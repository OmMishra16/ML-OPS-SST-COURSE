# Week 3 — LLM Serving & Inference Optimization; MLOps (Tracking, Registry, CI/CD, Drift)

Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week3/`):
- Session 1: `w3s1_slides.pdf`, `w3s1_pre_handout.pdf`, `w3s1_post_handout.pdf`, `w3s1_demo_llm_serving.ipynb`
- Session 2: `w3s2_slides.pdf`, `w3s2_pre_handout.pdf`, `w3s2_post_handout.pdf`, `w3s2_demo_mlops.ipynb`

Instructor: Priyansh Saxena (Scaler School of Technology, "ML System Design and LLMOps").
Course primary text referenced: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — Ch. 7 for Session 1; Ch. 6 & 8 for Session 2.

(note: some handout text lost en-dashes in extraction, e.g. "24×" in the handouts means "2–4×", "37%" means "3–7%", "7080%" means "70–80%", "Weeks 13" means "Weeks 1–3". These notes use the corrected ranges.)

---

## Session 3.1 — LLM Serving and Inference Optimization
*Subtitle: Memory Bounds • KV Cache • Quantization • Batching • Deployment*

Agenda (5 sections):
1. Why LLM Inference Is Hard
2. KV Cache and Memory Management
3. Quantization
4. Batching and Speculative Decoding
5. Deployment Choices

### TL;DR
- LLM inference is **memory-bound twice over**: weights (params × bytes/param; GPT-3 175B @ fp16 = 350 GB) and the **KV cache** (2 × L × H × d × S × bytes per request). Generation is **sequential** (autoregressive) and **memory-bandwidth bound, not compute-bound** — "the path to faster LLM inference goes through memory, not FLOPs."
- **KV cache** trades memory for compute (constant per-token cost instead of linearly growing). Naive pre-allocation causes **fragmentation**; **PagedAttention (vLLM)** pages the KV cache like OS virtual memory → up to 90% less fragmentation, ~24× throughput vs HuggingFace Transformers (~5 → ~120 req/s on an A100).
- **Quantization is the "almost free lunch"**: fp16 → INT8 → INT4 cuts memory 2–4× with small quality loss (INT8 <1% PPL, INT4 ~3–7% PPL). GPTQ (Hessian-based error compensation), AWQ (protect top 1% salient weights), GGUF/llama.cpp (CPU). INT4 70B = 35 GB → fits one 40 GB A100.
- **Continuous batching** raises GPU utilization from 20–30% (static) to 70–85% → 3–4× throughput. **Speculative decoding** (small draft model proposes K tokens, large model verifies in one parallel pass) → 2–3× speedup.
- **Deployment**: ONNX/ORT for portability, TensorRT for max NVIDIA speed, TFLite/CoreML for edge, vLLM/TGI for LLM serving. **API vs self-host** break-even rule of thumb: <$10K/month API spend → API; $10K–$50K → case-by-case; >$50K → almost always self-host.

### Concepts (in lecture order)

#### §1 Why LLM Inference Is Hard

##### 1.1 The memory problem: GPT-3 by the numbers
- GPT-3 has **175 billion parameters**. At fp16 (2 bytes/param): 175B × 2 = **350 GB** just to load weights.
- A top-of-the-line A100 has **80 GB VRAM** → **minimum 5× A100s just for weights**. Then add KV cache, activations, framework overhead…
- **Contrast with training**: in training you can accumulate gradients across batches; at inference you must hold the entire model in-flight for every request. "No gradient checkpointing trick helps here."
- **Formula: params × bytes/param = RAM**; fp32 = 4 B/param, fp16 = 2 B/param.

| Model | Params | fp16 RAM |
|---|---|---|
| GPT-2 | 1.5B | 3 GB |
| LLaMA-2-7B | 7B | 14 GB |
| LLaMA-2-13B | 13B | 26 GB |
| LLaMA-2-70B | 70B | 140 GB |
| GPT-3 | 175B | 350 GB |

##### 1.2 Sequential generation: why you can't just parallelize
- Autoregressive: token t1 is sampled; t2 depends on t1 (must wait); t_k depends on t1…t_{k−1}. **Strictly sequential**, unlike training batches.
- GPU is great at doing many things at once, but **each new token requires a full forward pass**.
- Even on an A100: **~20–50 tokens/second for a 7B model** → a **500-token response ≈ 10–25 seconds** single-threaded.
- Training vs inference parallelism: training processes a batch of e.g. 256 sequences in parallel (all known in advance, fully parallelisable). Inference: you can't know t_k before computing t_{k−1}; the sequential dependency is baked into the autoregressive formulation.
- **"The only escape"**: run multiple requests in parallel (**batching**). GPU utilisation comes from many requests, not from one long sequence.

##### 1.3 Memory-bandwidth bound, not compute-bound (roofline model)
- **Compute-bound**: bottleneck is FLOPs; more compute ⇒ faster.
- **Memory-bandwidth bound**: bottleneck is how fast you can move weights from **HBM** to compute units.
- For large LLMs, generating a single token requires **reading all weights once**.
- **A100 memory bandwidth: 2 TB/s. At 350 GB weights, that's ~175 ms just to load weights per token!**
- Implications for optimization:
  - Reducing FLOPs (pruning, distillation) helps **less than expected**.
  - Reducing model size in memory (**quantization**) helps **a lot**.
  - Improving memory access patterns (**fused kernels, FlashAttention**) helps **a lot**.
  - Buying a "faster GPU" helps less than buying one with **higher memory bandwidth**.
- **Key insight: the path to faster LLM inference goes through memory, not FLOPs.**

##### 1.4 The cost math: tokens × price × volume
Back-of-envelope, GPT-3.5 at scale:
- GPT-3.5 Turbo: **$0.002 per 1K tokens** (2023 pricing); avg response 1,000 tokens; 1M queries/day.
- Cost = 1M × 1K × $0.002/1K = **$2,000/day → $60,000/month → $730,000/year**.

Modern pricing (2024, as on slide):
- "GPT-4o: $15/1M input + $60/1M output tokens." At 10M queries/day, 1K tokens each: **$600K/day**. "This is why LLM inference optimization is a $1B engineering problem."
- (note: the demo notebook uses $2.50/1M input, $10/1M output for GPT-4o "mid-2025". The slide's $15/$60 figures are closer to GPT-4/o1-class pricing; use whichever figure the question states.)

**The 3 levers to pull costs down:**
1. **Fewer tokens** — prompt compression, caching, fewer retries.
2. **Cheaper model** — right-size the model to the task; "a 7B fine-tune often beats GPT-4 at a fraction of the cost."
3. **Self-host** — amortise GPU cost over volume; **break-even is roughly $10K/month API spend**.

#### §2 KV Cache and Memory Management

##### 2.1 What is the KV cache and why do we need it?
- Every attention layer computes Keys (K) and Values (V) for every token. Attention = softmax(QKᵀ/√d)·V. At token t you need K and V for all tokens 1…t−1.
- **Without KV cache**: generating token N requires re-computing K, V for all N−1 previous tokens; **cost per token grows linearly with sequence length**; in a 1000-token response the last token costs 1000× more to compute than the first.
- **With KV cache**: compute K, V for token t once and store them; when generating t+1, retrieve stored K, V; only compute K, V for the new token. **Cost per token: constant regardless of sequence length.** Standard in every production LLM serving system.
- **Trade-off: saves compute, costs memory.**

##### 2.2 KV cache memory: "the numbers are alarming"
**KV mem = 2 × L × H × d × S × bytes**
- L = number of layers, H = attention heads, d = head dimension, S = sequence length; factor 2 = one K tensor + one V tensor. (note: this is per request / per sequence; multiply by batch size/concurrent requests.)
- **GPT-3 (fp16)**: L = 96, H = 96, d = 128, S = 2048, 2 bytes → slide states "**KV cache ≈ 100 GB per request**; with 10 concurrent requests: **1 TB** of KV cache alone." (note: plugging these values into the formula gives ≈ 9.7 GB, not 100 GB; the slide/handouts consistently quote ~100 GB, so be aware of the discrepancy if asked to compute it.)
- Practical numbers (7B model, fp16):

| Item | Memory |
|---|---|
| Weights | 14 GB |
| KV cache per request (2K ctx) | ~2 GB |
| 10 concurrent requests | ~20 GB |

- **The real problem**: you can't pre-allocate KV cache for all possible requests. If you pre-allocate max-length buffers, most memory is wasted — **a 100-token request wastes 95% of its 2048-token allocation**. This is **memory fragmentation: the leading cause of low GPU utilisation in naive LLM serving.**

##### 2.3 PagedAttention: OS paging for the KV cache
- Insight from operating systems: in OS virtual memory, physical RAM doesn't need to be contiguous; allocate in fixed-size pages and map them to virtual addresses; pages can be anywhere in physical memory → no fragmentation.
- **PagedAttention (vLLM, 2023)**:
  - KV cache divided into fixed-size **blocks ("pages")**; each block holds K, V for a fixed number of tokens.
  - Blocks are **allocated on demand** as generation proceeds.
  - Blocks can be **shared across parallel beams** (beam search).
  - Result: **near-zero internal fragmentation**.
- What it achieves: memory fragmentation **reduced by up to 90%**; **24× higher throughput vs HuggingFace Transformers** on the same hardware; much larger batch sizes; **copy-on-write for shared prefixes (system prompts)**.
- Diagram (reconstructed): Contiguous (wasted): `req1 req1 req1 free free free` vs Paged (interleaved): `req1 req2 req1 req2 req1 req2`.

##### 2.4 vLLM: putting it together
- High-throughput LLM serving library from **UC Berkeley (Kwon et al., 2023)**. Core innovation: PagedAttention. Continuous batching built in. Supports most open-weight models (LLaMA, Mistral, Qwen…). **OpenAI-compatible API server out of the box.**
- Throughput on the same A100: **HuggingFace Transformers ~5 req/s vs vLLM ~120 req/s → 24×**.

| Feature | Benefit |
|---|---|
| PagedAttention | No KV waste |
| Continuous batching | Higher GPU util |
| CUDA kernels | Fused ops |
| Tensor parallel | Multi-GPU support |
| Prefix caching | Shared system prompts |

- **Production reality**: vLLM (or equivalents **TGI, SGLang**) is the default choice for self-hosted open-weight LLM serving. "If you are evaluating self-hosting, start here."

#### §3 Quantization

##### 3.1 Precision levels: fp32 to INT4
- **fp32** (full): 4 B/param. Gold standard. Used in training.
- **fp16** (half): 2 B/param. 2× smaller. Minor accuracy loss. **Standard for inference.**
- **bf16**: 2 B/param, **wider exponent range** than fp16 → better for **training stability**.
- **INT8**: 1 B/param. 4× smaller than fp32. Works well for most tasks.
- **INT4**: 0.5 B/param. 8× smaller. "Surprisingly good quality."
- **INT2**: 0.25 B/param. Usually degrades too much.

| Precision | Bytes | 70B RAM |
|---|---|---|
| fp32 | 4.0 | 280 GB |
| fp16 | 2.0 | 140 GB |
| INT8 | 1.0 | 70 GB |
| INT4 | 0.5 | 35 GB |
| INT2 | 0.25 | 18 GB |

**INT4: a 70B model fits on a single 40 GB A100 — "transformative for deployment cost."**

##### 3.2 GPTQ and AWQ: post-training quantization (PTQ)
**GPTQ (2022)**
- PTQ for GPT-style models; uses **Hessian-based second-order optimization**.
- Quantizes **layer-by-layer**; **compensates errors in remaining weights**.
- 4-bit GPTQ: 70B models on consumer GPUs (2× RTX 3090).
- Slightly slower than INT8 due to **dequantization overhead**.
- Why it works: naive rounding rounds each weight to nearest integer independently. GPTQ: after rounding w_i, **correct the error in w_{i+1}…w_n so the layer output is preserved**.

**AWQ (2023) — Activation-aware Weight Quantization**
- Not all weights matter equally. **Protect the top 1% of weights by activation magnitude** — these "salient" weights drive most of the output signal.
- Better perplexity than GPTQ on most benchmarks.

Benchmark — LLaMA-2-7B perplexity (lower = better):

| Method | PPL |
|---|---|
| fp16 baseline | 5.47 |
| GPTQ 4-bit | 5.85 |
| AWQ 4-bit | 5.60 |

AWQ is closer to fp16.

##### 3.3 GGUF: quantized models for CPU inference
- The CPU inference problem: most developers don't have A100s — can we run LLMs on a MacBook?
- **llama.cpp (Gerganov, 2023)**: C/C++ inference for LLaMA-family models; highly optimised for CPU SIMD (AVX2, ARM NEON); no Python, no CUDA, minimal dependencies.
- **GGUF format**: file format for quantized models (formerly **GGML**); **single self-contained file (weights + metadata + tokenizer)**; levels **Q2_K, Q3_K, Q4_K, Q5_K, Q8_0**; **Q4_K_M: 7B model = 4.1 GB, runs on MacBook Air M1**. Available for most open models on Hugging Face.

Speed (7B model, Q4):

| Device | Tok/s |
|---|---|
| MacBook Air M1 (CPU) | ~20–30 |
| MacBook Pro M3 Max | ~60–80 |
| RTX 4090 (GPU) | ~120–150 |
| A100 (fp16) | ~80–100 |

- Practical use case: **local dev, privacy-sensitive apps, edge deployment, offline demos**. **Not for high-throughput production (use vLLM).**

##### 3.4 Quantization trade-off: quality vs size vs speed

| Precision | Memory (70B) | Relative speed | Quality loss |
|---|---|---|---|
| fp32 | 280 GB | 1× baseline | None |
| fp16 | 140 GB | 1.8× | Negligible |
| INT8 | 70 GB | 2× | Very small (<1% PPL) |
| INT4 | 35 GB | 3× | Small (~3–7% PPL) |
| INT2 | 18 GB | 4× | Large (often unusable) |

- **The "free lunch" observation**: INT8/INT4 are "almost free" — small quality loss for 2–4× memory and speed gain. Most production systems run fp16; INT8 is becoming standard.
- **When quantization fails**: tasks requiring precise numerical reasoning; extreme INT2/INT3 levels; models not well-trained to begin with. **Always validate on your task, not just perplexity.**

(5-minute break slide: "Your GPU is still running. You, however, need a break.")

#### §4 Batching and Speculative Decoding

##### 4.1 Static vs continuous batching
**Static (naive) batching**
- Collect N requests, run them as a batch; simple: one big forward pass for all N.
- Problem: sequences have different lengths. If one request needs 2000 tokens and others need 10, **the short ones wait for the long one**; GPU idle while waiting for the slowest sequence.
- Real-world GPU utilization: **20–30%**.

**Continuous (iterative) batching**
- After **each decoding step**, check if any sequence is done; if sequence i finishes, **immediately insert a new request into that slot**.
- Batch composition changes at every step; no waiting — as soon as a slot is free, it fills.
- GPU utilization: **70–85%**.

**Result**: continuous batching ≈ **3–4× higher throughput** on the same hardware vs static. Default in vLLM, TGI (Text Generation Inference), and every serious LLM serving framework. (Post handout: zero quality cost.)

##### 4.2 Speculative decoding: the draft model trick
- Bottleneck recap: large model = slow per-token generation; we want more tokens/sec without quality loss. **Key insight: verifying tokens is cheaper than generating them.**
- **Algorithm:**
  1. **Draft**: small fast model proposes K tokens.
  2. **Verify**: large model processes all K tokens in **one forward pass** (parallel, not sequential).
  3. **Accept/reject**: if the large model agrees with draft token i, accept; **first disagreement ⇒ reject tokens i…K, resample from large model**.
  4. Repeat from step 1 with accepted tokens as context.
- **Why verification is parallel**: the large model computes P(t_i | t_1…t_{i−1}) for all i at once using **teacher-forcing** — it does not need to autoregressively wait between tokens (draft tokens are already known inputs).
- Performance: **speedup 2–3× for greedy decoding**. Best for **code generation, factual recall** (predictable outputs). Worst for **creative writing** (draft model diverges quickly). **Draft model must be same architecture family.**

##### 4.3 In-class Quiz (with answers)
**Q1 — KV cache trade-off.** 7B model on one A100 (80 GB); weights 14 GB fp16; each request 4096-token context; KV cache ≈ 4 GB/request. How many concurrent requests before KV cache exhausts VRAM? First optimization?
- **A1**: Available = 80 − 14 = 66 GB; ⌊66/4⌋ = **16 concurrent requests**. First optimization: **PagedAttention (vLLM)** — eliminates KV fragmentation (up to 90% waste in naive pre-allocation), letting the same 66 GB serve significantly more requests.

**Q2 — Quantization choice.** Production summarization on LLaMA-2-70B fp16 on 4× A100s, $40K/month. Evaluating INT4. Max A100 reduction? Quality risk to validate?
- **A2**: Slide answer: "INT4 halves memory (140 GB → 70 GB). Could reduce from 4× A100 to 1–2× A100, saving $20–30K/month." (note: per the precision table, INT4 is 35 GB — a quarter of fp16; 140 → 70 GB is INT8. Either way, 1–2 A100s is the answer; 35 GB fits a single 40/80 GB A100 with KV-cache headroom on the second.) Quality risks: (1) run summarisation benchmarks (**ROUGE, BERTScore**) on your production set, not just perplexity; (2) stress-test on your **longest documents**; (3) **A/B test with human evaluators** before full rollout.

#### §5 Deployment Choices

##### 5.1 ONNX and TensorRT: compilation for speed
**ONNX (Open Neural Network Exchange)**
- Intermediate representation for neural networks; export via `torch.onnx.export()`; run on any ONNX runtime (CPU, GPU, mobile, edge).
- Good for cross-platform deployment, model portability. Not always as fast as native PyTorch on GPU.

**ONNX Runtime (ORT)** — Microsoft's optimized ONNX inference engine
- Graph optimizations: **operator fusion, constant folding**. Backends: CPU, CUDA, DirectML, CoreML, TensorRT.
- Often **1.5–2× faster than raw PyTorch on CPU**.

**TensorRT (NVIDIA)** — optimizing compiler for NVIDIA GPUs
- Graph optimization: **fuse layers, eliminate redundant ops**.
- **Precision calibration**: auto INT8 with sample data.
- **Kernel auto-tuning**: best CUDA kernel per op shape.
- Speedup: **2–4× over PyTorch on same GPU**.

When to use each:

| Use case | Tool |
|---|---|
| Cross-platform | ONNX + ORT |
| Max GPU speed | TensorRT |
| Mobile/edge | TFLite / CoreML |
| LLM serving | vLLM / TGI |

##### 5.2 Self-hosted vs API: the decision framework

| Dimension | API (OpenAI, Anthropic, Cohere) | Self-hosted (Llama, Mistral) |
|---|---|---|
| Setup | Minutes. No infra. | Days–weeks. GPU cluster. |
| Cost | Pay-per-token. Expensive at scale. | Fixed GPU cost. Cheap at scale. |
| Data privacy | Data leaves your infra | Data stays on-prem |
| Customization | Limited (system prompt, fine-tune API) | Full control (RLHF, LoRA, etc.) |
| Latency control | You share capacity with all users | Dedicated hardware |
| Model quality | GPT-4 class (2024 frontier) | LLaMA-3-70B ≈ GPT-4 on many tasks |
| Engineering burden | Near zero | Significant (infra, monitoring, updates) |

- **Use API when**: prototyping, low volume (<$10K/month), no sensitive data, need frontier model quality.
- **Self-host when**: high volume (>$10K/month API spend), data sensitivity, need fine-tuning, latency SLA.

##### 5.3 Cost comparison at scale: API vs self-hosted
Scenario: **100M output tokens/day**.
- **GPT-4o API**: output $60/1M → 100M × $60/1M = **$6,000/day → $180,000/month**.
- **Self-hosted LLaMA-3-70B on 8×A100**: instance ~**$25/hour**; throughput **~2,000 tokens/sec** with continuous batching; 100M tokens/day ≈ **~14 hours of GPU time**; cost 14 × $25 = **$350/day → ≈ $10,500/month**.
- **Savings: 17× cheaper at this volume.**

Break-even analysis:
- Below ~$10K/month API → **use API** (engineering cost too high).
- $10K–$50K/month → **evaluate case-by-case**.
- Above $50K/month API spend → **almost always self-host**.

**Hidden self-hosting costs**: 1–2 ML engineers to maintain the stack; model updates (new LLaMA versions, safety patches); monitoring, autoscaling, HA infrastructure; reserved vs spot GPU instances (**2–3× cost difference**).

##### 5.4 Summary slide ("What We Covered Today")
- §1: 175B params = 350 GB fp16; can't parallelize tokens; memory-bandwidth bound; cost = tokens × price × volume.
- §2: KV cache stores K,V for all previous tokens; up to 100 GB/request at scale; PagedAttention; vLLM 24×.
- §3: fp32 → fp16 → INT8 → INT4; GPTQ, AWQ (PTQ); GGUF (llama.cpp); quality/size/speed trade-offs.
- §4–5: continuous batching 70%+ util; speculative decoding 2–3×; ONNX/TensorRT; API vs self-host framework.
- Closing joke: "Serving a 175B model on a MacBook. Context length optional. What could go wrong?"

### Demo notebook — `w3s1_demo_llm_serving.ipynb`
Framing: "inference is compute-bound not IO-bound, and the cost structure means every optimization decision has direct dollar implications." (note: the notebook intro says "compute-bound", but the slides' key point is that LLM decoding is **memory-bandwidth bound**; go with the slides for exam answers.) Four parts:

**Part 1 — LLM memory calculator** ("Can we fit this model on an A100?")
```python
def llm_memory_gb(params_billions, dtype="fp16"):
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    return params_billions * 1e9 * bytes_per_param[dtype] / 1e9
```
GPU reference: A100-40GB, A100-80GB, H100-80GB. Output table (weights only):

| Model | Params | fp32 | fp16 | int8 | int4 |
|---|---|---|---|---|---|
| GPT-3 | 175B | 700 GB | 350 GB | 175 GB | 87.5 GB |
| Llama-7B | 7B | 28 | 14 | 7 | 3.5 (fits RTX 4090 24GB) |
| Llama-70B | 70B | 280 | 140 | 70 | 35 (fits single A100-40GB) |
| Llama-3-8B | 8B | 32 | 16 | 8 | 4.0 (fits RTX 4090) |
| Mistral-7B | 7B | 28 | 14 | 7 | 3.5 (fits RTX 4090) |

Rules of thumb printed: weights only — **add ~20% for activations and KV cache during inference**; **for training add ~8×** (fp32 optimizer states + gradients + activations).

**Part 2 — Cost estimator: API vs self-hosted** (build-vs-buy break-even)
- `compute_llm_costs(requests_per_day, avg_input_tokens, avg_output_tokens, api_input_cost_per_1m, api_output_cost_per_1m, gpu_hourly_rate, gpu_active_hours_per_day)`.
- Break-even assumes **70/30 input/output token split**: `effective_cost_per_token = (0.7*in_price + 0.3*out_price)/1e6`; `breakeven_tokens_per_month = monthly_gpu_cost_optimized / effective_cost_per_token`.
- Worked example: 10,000 req/day, 500 in / 200 out tokens; GPT-4o $2.50/1M in, $10/1M out; GPU = **AWS p4d.24xlarge (8× A100-80GB) at $32.77/hr**; optimized = 12 active hours/day ("scale down at night").
- Resulting numbers (computed from the code): 7M tokens/day, 210M/month; API **$32.50/day → $975/month → ~$11,862/year**; GPU always-on **$23,594/month (~$283K/yr)**; optimized 12h **$11,797/month (~$141.6K/yr)**. Utilization vs theoretical (assumes ~40K tokens/s for Llama-70B fp16) ≈ **0.2%**. Conclusion: **"At current 10,000 req/day: API is cheaper."**
- (note: with the code's formula, break-even ≈ 2.48B tokens/month ≈ ~118K req/day; the closing discussion text says "~42K req/day". Also the print label says "1x A100-80GB" while the rate is for the 8×A100 p4d instance. Treat the qualitative conclusion as the takeaway.)
- Other factors printed: **API** — data privacy, rate limits, vendor dependency, no cold start. **Self-hosted** — full control, lower latency, compliance, ops burden.

**Part 3 — Quantization demo (DistilBERT fp32 vs fp16, CPU)**
- `DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16)`; params counted via `sum(p.numel() ...)`; ~66M params → ~265 MB fp32 vs ~133 MB fp16 (**50% / 2× smaller**).
- Benchmarks 100 forward passes (5 warmup) with `time.perf_counter()` under `torch.no_grad()`.
- Takeaways: **on GPU fp16 is typically 2–4× faster because tensor cores are optimized for fp16 matmul; on CPU the improvement is modest** (no tensor cores). (note: on many CPUs fp16 can even be slower.) Output comparison: max abs/relative diff of `last_hidden_state` tiny → "effectively identical outputs — quantization is nearly lossless here."

**Part 4 — KV cache illustration** (simulation: 12 layers, 8 heads, head_dim 64, prompt 256 + gen 256 = 512 tokens)
- Without cache: K,V computations = Σ_{i<256}(256+i) = **98,176**; with cache: **512** (one per token) → **~192× fewer K,V computations** (simulated 9.8 s vs 0.05 s at 0.1 ms/op).
- KV memory per token = 2 × n_layers × n_heads × head_dim × 2 bytes = **24 KB/token**; 512-token seq = **12 MB**; 8 users ≈ 96 MB; 100 users ≈ 1.2 GB; **128K-token context ≈ 3 GB per request** (for this tiny model).
- Insight: **"the KV cache trades memory for compute. It's the reason batch size is limited under high-concurrency serving."**
- "Why this matters for system design":
  - KV cache is **non-negotiable** for production; every serious framework (vLLM, TensorRT-LLM, TGI) implements it.
  - **PagedAttention** manages KV cache like OS virtual memory — fixed pages, sharing across parallel beams → vLLM serves **10–20× more concurrent requests** than naive implementations.
  - A **32K-token document can consume several GB** of KV cache for a single request — the dominant memory consumer in long-context serving.
- **Discussion prompt**: what would you need to build beyond the GPU server to hit the break-even utilization reliably? → think **autoscaling, queue management, cold start**.

### Pre/Post handout extras
**Pre-handout**
- Framing: "deploying an LLM at scale is an engineering problem measured in hundreds of gigabytes of memory and thousands of dollars per day."
- Session plan: opening — estimate memory cost of your last LLM project; attention walk-through; KV cache from first principles; PagedAttention/vLLM; precision table; GPTQ/AWQ; static vs continuous batching; speculative decoding step by step; self-host vs API at 100M tokens/day; two quiz questions.
- Prep: **cost estimate exercise** — pick an LLM project (e.g., customer support bot, coding assistant); estimate tokens/request, requests/day; at **$0.002/1K tokens** what's the monthly cost? Recall self-attention (QKV, formula) — understanding *why K and V must be stored* is key. Optional: read vLLM PagedAttention blog; install llama.cpp and run a Q4 model locally. No coding environment needed (architecture & trade-offs focus). Duration ~2 h; prereqs: Transformers, Weeks 1–2.

**Post-handout key takeaways**: (1) memory-bandwidth bound — quantization and fused kernels attack the right bottleneck; (2) KV cache is the biggest memory hog in production; (3) quantization (GPTQ/AWQ) cuts memory 2–4× with only ~3–7% perplexity increase — always validate on your task; (4) static batching leaves GPUs **70–80% idle**; continuous batching → 70–85% util, 3–4× throughput at zero quality cost.

**Test Your Understanding (with handout answers)**
- **Q1** 7B fp16 = 14 GB → INT4 AWQ? **A**: 7B × 0.5 = **3.5 GB** (4× reduction); small perplexity increase (~3–7% for well-tuned AWQ 4-bit), usually imperceptible for summarization/instruction-following — always evaluate on your task.
- **Q2** Why does KV cache cause fragmentation, and how does PagedAttention fix it? **A**: Naive systems pre-allocate a contiguous buffer for max context per request; most requests use a fraction → wasted, unavailable to others (**internal fragmentation**). PagedAttention divides KV cache into fixed-size blocks allocated **on demand**, so memory is only consumed for tokens actually generated.
- **Q3** 5M calls/day × 800 output tokens at $0.060/1K output tokens → monthly? **A**: 4×10⁹ tokens/day → 1.2×10¹¹/month → **$7,200,000/month**. Far above $10K/month → self-hosting almost certainly worth it; even at $1M/month self-host cost you save $6.2M.
- **Q4** Why can the verifier process K draft tokens in one pass? **A**: draft tokens are already known → teacher-forcing computes P(t_i | t_1…t_{i−1}) for all positions in one batched pass (like training); no sequential dependency because inputs are provided, not generated.

**Exercises**
- **Ex 1 — Memory footprint for LLaMA-2-70B**: (A) weights fp16 and INT4; (B) KV cache with 80 layers, 64 heads, head dim 128, 4096 ctx, fp16 via 2×L×H×d×S×bytes; (C) concurrent requests on 8×A100 (640 GB) in fp16 vs INT4.
  - (note: worked numbers — A: 140 GB fp16, 35 GB INT4. B: 2×80×64×128×4096×2 ≈ 10.7 GB/request. C: fp16 ⌊(640−140)/10.7⌋ ≈ 46; INT4 ⌊(640−35)/10.7⌋ ≈ 56. Real LLaMA-2-70B uses grouped-query attention with 8 KV heads, which would cut KV ~8×, but the exercise uses 64.)
- **Ex 2 — Serving stack for 10M LLM calls/day, p99 200 ms**, avg 200 input / 500 output tokens. Cover: model choice + quantization level; serving framework (vLLM/TGI — justify); batching strategy; #GPUs with throughput math; one monitoring metric to alert on. Hint: **vLLM on one A100 at INT4 ≈ 2,000 tokens/sec with continuous batching**.
  - (note: 10M × 500 = 5B output tokens/day ≈ 57.9K tok/s average → ~29 A100s before peak headroom/replication. A 200 ms p99 cannot cover full 500-token generation, so interpret it as time-to-first-token / per-step latency and state the assumption.)
- **Challenge — GPT-4o API vs self-hosted LLaMA-3-70B at 100M output tokens/day**: A = $60/1M output; B = 8×A100 at $25/hr, INT4 + continuous batching ~4,000 tok/s, +20% overhead, 0.5 FTE at $200K/yr. Compute monthly costs and break-even.
  - (note: A = $180K/month. B: 100M/4,000 = 25,000 s ≈ 6.94 h/day ×1.2 ≈ 8.33 h × $25 ≈ $208/day ≈ $6,250/month + engineering $8,333/month ≈ **$14.6K/month**. Self-host variable cost ≈ $2.08 per 1M tokens vs $60 API → break-even ≈ 8,333/(60−2.08) ≈ 144M tokens/month ≈ ~4.8M tokens/day; above that self-host wins, assuming you pay only for GPU hours used.)

**Further reading**: Kwon et al. 2023 (PagedAttention, SOSP — intro + §3); Frantar et al. 2022 (GPTQ, arXiv 2210.17323 — §2 OBQ intuition, error compensation); Lin et al. 2023 (AWQ, arXiv 2306.00978 — §3.1 salient weights); Chip Huyen Ch. 7. Online: vLLM docs, llama.cpp GitHub, HF TGI docs, Tim Dettmers blog (INT8/INT4), Lilian Weng "LLM Inference Optimization" (2023), Karpathy "Let's build the GPT Tokenizer".

### Likely doubts & exam angles
1. **Why is LLM decoding memory-bandwidth bound?** Each token needs all weights read from HBM once; e.g. 350 GB / 2 TB/s ≈ 175 ms per token for GPT-3 on A100 bandwidth. So quantization and better memory access (FlashAttention, fused kernels) help more than FLOP reductions or "faster" GPUs.
2. **Weights memory formula?** params × bytes/param (fp32 4, fp16/bf16 2, INT8 1, INT4 0.5). 70B: 280/140/70/35 GB.
3. **KV cache formula and what each term is?** 2 × L × H × d × S × bytes; 2 = K and V. Grows linearly with sequence length and with number of concurrent requests.
4. **Concurrency on a GPU?** (VRAM − weights) / KV-per-request; e.g. ⌊(80−14)/4⌋ = 16. First fix: PagedAttention; also quantize weights (and KV).
5. **PagedAttention vs naive allocation?** Naive = contiguous max-length buffers → 100-token request wastes 95% of a 2048 slot. Paged = fixed blocks allocated on demand, non-contiguous, sharable (beam search, copy-on-write prefixes) → up to 90% less fragmentation, 24× throughput.
6. **Static vs continuous batching?** Static waits for the longest sequence (20–30% util); continuous refills slots after every decode step (70–85% util, 3–4× throughput).
7. **When does speculative decoding help/hurt?** Helps predictable outputs (code, factual recall), 2–3× for greedy; hurts creative writing (low acceptance). Draft model must share architecture family (tokenizer/vocab). Output distribution is preserved because the big model verifies.
8. **GPTQ vs AWQ?** GPTQ: Hessian-based, layer-by-layer error compensation into remaining weights. AWQ: protect top 1% salient weights by activation magnitude; better PPL (5.60 vs 5.85 vs fp16 5.47 on LLaMA-2-7B).
9. **When to choose GGUF/llama.cpp vs vLLM vs TensorRT vs ONNX?** GGUF: local/edge/privacy/offline, CPU; vLLM/TGI: high-throughput LLM serving; TensorRT: max NVIDIA GPU speed (2–4×); ONNX+ORT: cross-platform portability (1.5–2× on CPU).
10. **API or self-host?** <$10K/month → API; $10K–$50K → case-by-case; >$50K → self-host. Account for hidden costs (1–2 engineers, updates, monitoring/HA, reserved vs spot 2–3×) and non-cost factors (privacy, customization, latency SLA, frontier quality).

---

## Session 3.2 — MLOps: Experiment Tracking, Model Registry, and CI/CD for ML
*Subtitle: Reproducibility • Registry Lifecycle • CI/CD Gates • Distribution Shift*

Agenda: (1) Experiment Tracking, (2) Model Registry, (3) CI/CD for ML, (4) Distribution Shift and Monitoring.

### TL;DR
- **Track everything from day one** (loss curves, all metrics, hyperparameters, data version, git commit, artefacts, system metrics, sample predictions). MLflow / W&B = your ML lab notebook. Reproducibility checklist: seeds, pinned versions, DVC, git hash, Docker.
- **A model registry is not a file store**: semantic version, lineage, lifecycle (**Training → Staging → Production → Archived**), audit trail, **one-command rollback**. Promotion via explicit rules (e.g., beat prod NDCG@20 by ≥1%, no slice regression, p99 ≤ 100 ms, diversity drop ≤ 5%).
- **ML CI/CD tests code + data + model**: pipeline = Code push → Unit tests → Data validation → Model training → **Perf gate** (block if new ≤ prod) → Staging → **Canary (10%)** → Full production. Triggers: code push, new data, drift alert, manual.
- **Three shift types**: covariate (P(X)), label (P(Y)), concept drift (P(Y|X)). **~80% of drift alerts are human/pipeline errors** — "fix your pipeline before blaming the world."
- **Monitoring has 4 layers** (data quality → feature drift → prediction drift → business metrics); **three retraining triggers** (scheduled, drift-triggered, performance-triggered), chosen per domain.

### Concepts (in lecture order)

#### §1 Experiment Tracking

##### 1.1 The problem: ML without a lab notebook
"Sound familiar?"
- "Did we try dropout = 0.3? What was the result?"
- "Which dataset version did the best model train on?"
- "Why did experiment 47 beat experiment 46?"
- "Can you reproduce last month's model exactly?"

**The cost of not tracking**: you cannot reproduce your best result, cannot explain what changed, cannot hand work to a teammate — "starting from scratch every time."

**What to track per experiment:**
- Loss curves — train, val, test
- All metrics — accuracy, F1, AUC, NDCG
- Hyperparameters — every knob you turned
- **Data version — DVC tag or dataset hash**
- **Code version — git commit hash**
- Artefacts — model file, preprocessor, config
- System metrics — GPU memory, training time
- Sample predictions — for sanity checks

##### 1.2 MLflow: your ML lab notebook
Core concepts:
- **Experiment**: a named collection of runs (e.g. "fraud-detection-v2").
- **Run**: one execution of your training script.
- **Parameters**: inputs — hyperparameters, config.
- **Metrics**: outputs — logged over time (loss per epoch).
- **Artefacts**: files — model, plots, confusion matrices.
- **Tags**: free-form metadata (git hash, author, dataset name).

Minimal MLflow tracking (slide):
```python
import mlflow
mlflow.set_experiment("fraud-v2")
with mlflow.start_run():
    mlflow.log_params(config)
    for epoch in range(N):
        mlflow.log_metric("val_auc", auc, step=epoch)
    mlflow.sklearn.log_model(clf, "model")
    mlflow.set_tag("git_commit", get_git_hash())
```
Tools: **MLflow (open source), Weights & Biases (hosted), Neptune, Comet**.

##### 1.3 Reproducibility: why it's hard and how to fix it
Sources of nondeterminism:
- **GPU parallel ops** — CUDA ops not deterministic by default; set `torch.use_deterministic_algorithms(True)`.
- **Floating-point rounding** — differs across hardware and library versions.
- **Random seeds** — data shuffling, weight init, dropout.
- **Library version drift** — PyTorch 2.0 vs 2.1 may give different results.
- **Data mutation** — if training data changes after the run, you can't reproduce.

**Reproducibility checklist:**
- ✓ Set all random seeds (torch, numpy, random, `os.environ["PYTHONHASHSEED"]`)
- ✓ Pin library versions (requirements.txt or conda env)
- ✓ Version data with DVC and log the tag
- ✓ Log git commit hash with every run
- ✓ Use Docker for environment reproducibility

Tools: **DVC** (data versioning), **Docker** (environment), **MLflow** (experiment context).

#### §2 Model Registry

##### 2.1 Registry vs S3: "A Tale of Two Artefact Stores"
**The S3 approach**: `fraud_model_v2.pkl`, `fraud_model_v3.pkl`, `fraud_model_v3_final.pkl`, `fraud_model_v3_final_FINAL.pkl`, `fraud_model_PROD_USE_THIS.pkl`.
→ **No versioning. No lineage. No lifecycle. No rollback. No audit trail.** When something breaks: "Which one is live? Who deployed it? When?"

**The registry approach** (example record):

| Property | Value |
|---|---|
| Version | v3.2.1 |
| Stage | Production |
| Trained on | dataset-2024-11 |
| Git commit | a3f91bc |
| Val AUC | 0.9847 |
| Promoted by | psaxena |
| Promoted at | 2024-11-15 14:32 |
| Previous | v3.1.0 (Archived) |

Post-handout: "The moment you have more than two models in production, you need a registry."

##### 2.2 Model registry lifecycle
Flow (reconstructed diagram):
```
Training --auto-register--> Staging --passes gate--> Production --new version--> Archived
                            (Automated eval:         (Serving live      (Kept for rollback
                             must beat prod model)    traffic)           & audit)
              Archived --rollback--> Production
```
**Promotion rules (example):**
- New model must beat production **NDCG@20 by ≥ 1%**
- **No regression on critical slices** (high-value users)
- **Latency p99 must stay ≤ 100 ms**
- **Diversity score must not drop > 5%**

**Rollback in one command** (MLflow client):
```python
# Demote bad model
client.transition_model_version_stage("fraud-detector", "3", "Archived")
# Promote previous
client.transition_model_version_stage("fraud-detector", "2", "Production")
```
(note: newer MLflow versions deprecate stages in favour of model version aliases, e.g. `@champion`; the concept is identical.)

(5-minute break: "Your models can train without you.")

#### §3 CI/CD for ML

##### 3.1 CI/CD for ML: beyond code testing
Standard software CI/CD: unit tests (does the code work?), integration tests (do components talk?), build (does it compile/package?), deploy to staging then production.

**ML adds three new concerns:**
1. **Data validation** — is the incoming data still valid?
2. **Model evaluation** — is the new model actually better?
3. **Training regression** — does training still converge correctly?

ML-specific test examples — **Data validation (Great Expectations)**:
- Schema: expected columns, dtypes
- Null rate per column ≤ threshold
- Value range checks (age ∈ [0, 130])
- **Volume: sudden drop = pipeline failure**
- Distribution: KL divergence vs baseline

**Model performance gate**: new model must beat production on a held-out eval set **before any deployment proceeds**.

##### 3.2 The ML CI/CD pipeline
```
Code Push → Unit Tests → Data Validation → Model Training → Perf. Gate → Staging Deploy → Canary (10%) → Full Production
                         (schema,                          (block if new
                          distribution)                     model ≤ prod)
```
**What triggers the pipeline?** Code push to main; new data batch available (scheduled); **drift alert from monitoring**; manual trigger (on-demand retraining).

**Tools:**
- Orchestration: **Airflow, Kubeflow Pipelines, Prefect**
- Data validation: **Great Expectations, dbt tests**
- CI trigger: **GitHub Actions, Jenkins**
- Environment: **Docker + Kubernetes**

##### 3.3 Quiz 1 (with answers)
**Q1 — The promotion gate.** Prod fraud model val AUC 0.9847; new model 0.9851 on the same test set. Auto-promote? Extra checks? (Hint: slices, latency, overfitting to eval set.)
- **A1**: **Not automatically — +0.0004 AUC is within noise.** Require: (1) **Sliced performance** — high-value users, rare fraud patterns, underrepresented demographic groups (a model can improve on average while regressing on a critical slice); (2) **Latency** — p99 ≤ 100 ms (more complex model may breach SLA); (3) **Eval set freshness** — if the held-out set is months old, a small gain may reflect overfitting to the old distribution.

**Q2 — The broken pipeline.** Refactored feature computation; unit tests pass; perf gate blocks: AUC 0.9847 → 0.8312. Debug order?
- **A2**: (1) **Feature computation diff** — did any feature value change, even one column? Print 10 rows old vs new pipeline and compare. (2) **Data leakage direction** — did you accidentally *remove* leakage (old pipeline leaked future data, inflating AUC)? (3) **Scaling stats** — still fitting scaler on training data only, or now on all data? (4) **Git diff** of the feature code to find the exact change.

#### §4 Distribution Shift and Monitoring

##### 4.1 The three types of distribution shift

| Type | Definition | Real-world example (slide) |
|---|---|---|
| **Covariate shift** | P(X) changes; P(Y\|X) stays the same | Credit scoring model trained pre-COVID; post-COVID income distributions are completely different |
| **Label shift** | P(Y) changes; P(X\|Y) stays the same | Spam filter: spammer tactics evolve; the spam category contains different emails each year |
| **Concept drift** | P(Y\|X) changes; same input, different correct output | "Wuhan" search queries in 2019 = travel information; in 2020 = COVID origin. Same input, completely different user intent |

(note: the slide's spam example reads more like a change in P(X|Y); the post-handout's label-shift example — a diagnosis model moved to a hospital with different disease prevalence — is the cleaner textbook case.)

**"The uncomfortable truth about drift alerts"**: a monitoring company CTO: "In my estimate, **80% of the drifts captured by our service are caused by human errors**" — bugs in data pipelines, missing values incorrectly inputted, wrong feature statistics used for scaling, **training-serving skew**. **Fix your pipeline before blaming the world.**

##### 4.2 Monitoring: four layers
- **Layer 1 — Data quality (check first)**: schema violations (unexpected column, wrong dtype); null rate spikes per feature; volume drops (pipeline failure signal); value range violations.
- **Layer 2 — Feature drift**: input distribution vs training baseline; statistical tests: **KS test, PSI (population stability index), Jensen-Shannon divergence**; tools: **Evidently AI, WhyLabs, Arize**.
- **Layer 3 — Prediction drift**: output distribution shift — classification: predicted class distribution; regression: predicted value distribution. Alert if it diverges significantly from deployment baseline.
- **Layer 4 — Business metrics**: CTR, conversion rate, revenue. **Slow to respond but carry the most signal.** Alert: **5% drop week-over-week in primary KPI**.

**Alert ordering matters**: Data quality alert → **fix pipeline**. Feature drift alert → **consider retraining**. Business metric drop → **urgent investigation**. **Don't wait for Layer 4.**

##### 4.3 Retraining triggers
1. **Scheduled retraining** — retrain weekly/monthly regardless of drift. Simple to implement and reason about. Risk: too infrequent for fast-moving domains (news, finance); too frequent = wasted compute for stable domains (medical imaging).
2. **Drift-triggered retraining** — statistical test fires when P(X) diverges beyond threshold. More adaptive but can fire on **false alarms (80% are pipeline bugs!)** → **must have a pipeline-bug checklist that runs first**.
3. **Performance-triggered retraining** — business metric drops below threshold (e.g., **CTR drops > 5% week-over-week**). **Most trustworthy signal but slowest to detect**; requires ground-truth labels — not always available immediately.

Choosing your strategy:

| Domain | Strategy |
|---|---|
| News recommendation | Drift-triggered |
| Fraud detection | Performance-triggered |
| Medical imaging | Scheduled (quarterly) |
| Credit scoring | Scheduled + regulatory |

##### 4.4 Quiz 2 (with answers)
**Q3 — Diagnosing a production drop.** RecSys CTR down 8% over two weeks; latency & errors unchanged; feature drift alerts on `last_session_duration`, `days_since_last_login`, `purchase_count_30d`. Process? Bug vs genuine drift? When retrain?
- **A3**: **Step 1: check for pipeline bugs first (80% of drift alerts are bugs).** Were these features recently changed in ETL? Check null rates (sudden spike = missing upstream data), value range changes (e.g., `purchase_count_30d` returning 0 for everyone = query bug), timestamp errors (wrong date window). **Step 2: confirm genuine drift** if no bug — compare week-over-week distributions with PSI or KS. **Retrain when**: pipeline healthy + drift statistically confirmed + a retrain on recent data shows improved offline metrics on a **fresh held-out set**.

**Q4 — Choosing a retraining strategy** for (a) real-time fraud detection, (b) quarterly employee performance review model.
- **A4**: **Fraud** — most likely **concept drift** (fraudsters adapt); **performance-triggered** — monitor precision/recall on flagged transactions daily, retrain when recall drops below threshold. Main risk: **delayed ground truth** (chargebacks take days) → may retrain on incomplete labels. **Employee review** — most likely **covariate shift** (workforce demographics change slowly); **scheduled quarterly**, aligned with review cycles. Main risk: too infrequent if the company undergoes rapid restructuring/hiring.

##### 4.5 Summary slide
- §1: track params, metrics, data version, code version, artefacts; MLflow/W&B; checklist: seeds, versions, Docker.
- §2: Registry ≠ S3 bucket; Staging → Production → Archived; lineage, audit trail, one-command rollback.
- §3: ML adds data validation + model perf gates; promote only if it beats production; code push → production.
- §4: 3 shift types; 80% of apparent drift = pipeline bugs; 4-layer monitoring (data → features → predictions → business); 3 retraining triggers.
- Closing: **"An untracked experiment is a forgotten experiment. An unmonitored model is a ticking clock."**

### Demo notebook — `w3s2_demo_mlops.ipynb`
Framing: "A model that works today but can't be reproduced, monitored, or safely updated isn't production-ready." Deps: `mlflow great-expectations scipy`.

**Part 1 — MLflow experiment tracking** ("Which run produced the model in production?" — answerable in 10 seconds)
- Breast-cancer dataset (`load_breast_cancer`), 80/20 split, 3 RandomForest variants: (50, depth 5), (100, depth 10), (200, None).
- Key patterns:
  - `mlflow.set_tracking_uri("sqlite:///mlflow.db")` — pin an explicit tracking URI; MLflow 3.x file backend (`./mlruns`) is deprecated, and implicit defaults cause the classic "my runs don't show up" (kernel cwd ≠ where `mlflow ui` runs).
  - `mlflow.set_experiment("breast_cancer_classifier")`; `with mlflow.start_run(run_name=f"rf_variant_{i}") as run:`
  - Wrap **preprocessing + model in a sklearn `Pipeline`** (StandardScaler + RF) so the logged/registered artefact reproduces the **full inference path** (guards against training-serving skew).
  - `mlflow.log_param(...)` (model_type, n_estimators, max_depth as str, random_state, train/test size); `mlflow.log_metric("accuracy"/"f1_score"/"roc_auc", ...)`; `mlflow.sklearn.log_model(pipeline, artifact_path="model")`.
  - Best run selected by AUC; UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.
- "What MLflow gives you": **reproducibility, comparison, artifact lineage, model registry** (Staging → Production with approval workflows).
- Key discipline: **"log everything before you need it."** The run that broke production 3 months ago is only diagnosable if you logged parameters at training time.

**Part 2 — Data validation with Great Expectations** ("first line of defense against silent model degradation")
- Implements a minimal GE-style `DataExpectation(column, check_fn, description)` (in prod: `ge.from_pandas(df)` + expectation suites).
- Suite derived from training stats: `user_age` not null & in [18, 100]; `session_length`, `purchase_count`, `avg_order_value` min ≥ 0; `days_since_last_purchase` in [0, 3650].
- Clean batch: 6/6 pass. Corrupted 200-row prod batch with three injected issues: **15 nulls in user_age (sensor dropout)**, **8 negative avg_order_value (refund processing bug)**, **5 days_since_last_purchase = 9999 (Y2K-style source bug)** → failures reported.
- Action: **"batch rejected. Model prediction pipeline NOT called. Alert sent to #data-quality Slack channel."**

**Part 3 — Distribution shift detection (KS test + PSI)**
- Data validation catches structural problems; **distribution shift is subtler: structurally valid, statistically different.**
- Scenario: e-commerce; train = Q1 (non-holiday), prod = **Q4 holiday season** (users younger, buy more, spend more). n = 5,000 each. Shifted: user_age (N(40,12) → N(33,10)), purchase_count (Poisson 2.5 → 5.5), avg_order_value (lognormal μ 4.0 → 4.5); similar: session_length, days_since_last_purchase.
- **KS test** (`scipy.stats.ks_2samp`): higher statistic = more different; **p < 0.05 = significant shift**.
- **PSI** — bins from training-distribution percentiles (10 bins), epsilon 1e-6 to avoid log(0):
  `PSI = Σ (prod% − train%) × ln(prod% / train%)`
- **PSI thresholds: < 0.10 no significant change; 0.10–0.25 moderate — investigate; > 0.25 major shift — model likely needs retraining.** (PSI "used heavily in credit risk".)
- (note: re-running the code gives PSI ≈ user_age 0.47, purchase_count 1.88, avg_order_value 0.37 → RETRAIN; session_length ≈ 0.01, days ≈ 0.02 → stable.) Recommendation printed: **TRIGGER RETRAINING with recent production data.**

**Part 4 — Retraining trigger simulation**
- 30 days of accuracy: days 1–10 stable ~0.93; days 11–25 gradual drift (−0.006/day); days 26–30 new model after retraining (recovering from 0.87 upward).
- Monitoring logic: **THRESHOLD = 0.85**, rolling **WINDOW_SIZE = 3** (smoothing, via `collections.deque(maxlen=3)`), **ALERT_CONSECUTIVE = 3** consecutive below-threshold days → alert: "schedule retraining job with data from last 30 days", notify ml-ops email + #ml-alerts Slack. Statuses: OK → BELOW_THRESHOLD (k/3) → RETRAINING_QUEUED → OK (new model deployed).
- (note: with seed 42 only days 24–25 fall below 0.85 (0.839, 0.837), so the strict 3-consecutive rule would not actually fire in a re-run; the intended story is alert near day 25 → retrain → recovery. A good discussion point: consecutive-day rules trade false alarms for detection delay.)
- **The monitoring stack in practice**:
```
Ground truth labels arrive → compute metrics → write to time-series DB
  → monitoring service reads → applies rolling window + threshold
  → fires alert → triggers retraining job
```
- Tricky part: **ground truth labels are often delayed** (loan default: ~90 days). Hence also monitor **proxy metrics available immediately — feature distributions, prediction distributions**.
- **Discussion**: name two label-free proxy metrics for early warning → e.g., **prediction/score distribution drift (Layer 3)** and **input feature drift via PSI/KS (Layer 2)**; also data-quality signals (null rate, volume) and model confidence. (note: answer drawn from the 4-layer monitoring slide.)

### Pre/Post handout extras
**Pre-handout**
- Framing: running hundreds of experiments without tracking = no lab notebook. `model_v3_final_FINAL.pkl` is not a production artefact. "Software CI/CD tests code. ML CI/CD tests code **and** data **and** model performance."
- Session plan: live MLflow demo; registry promotion walkthrough; CI/CD diagram (code push → production); GE validation gates; COVID search-query shift examples; retraining trigger design exercise; quiz — debug a degraded model.
- Prep: could you reproduce your best result from scratch / explain what changed between experiment 12 and 13? Recall **training-serving skew (Week 2)** — "this session explains one of its root causes." Optional: MLflow quickstart; Chip Huyen Ch. 8 opening story (grocery store prediction model that slowly went wrong). Duration ~2 h; prereqs Weeks 1–3 Session 1.

**Post-handout key takeaways**: track everything from day one ("Did we try dropout?" should never be an open question); registry ≠ file store (needed once >2 models in prod); CI/CD adds data and model tests — schema validation, distribution checks, performance gate — "without these gates, you are manually babysitting every deployment"; distribution shift is mostly catching your own bugs.

**Test Your Understanding (with handout answers)**
- **Q1** Reproduce experiment 27 of 50 three months later? **A**: exact dataset (version/snapshot), code (git commit), all hyperparameters, random seed, environment (Python + library versions). MLflow tracks these via `log_params()`, `log_metrics()`, `log_artifact()`, `set_tag("git_commit", ...)`. Without this, experiment 27 is effectively lost.
- **Q2** Problems with `fraud_model_v2_latest.pkl` in a 6-person team? **A**: (1) **No lineage** — can't know what data/code produced it; (2) **No lifecycle management** — "latest" means nothing; no Staging vs Production; (3) **No rollback path** — overwriting "latest" loses the previous model. Registry: lineage stored with model, stages control promotion, previous versions archived & restorable.
- **Q3** Fraud model, flagged-transaction CTR down 12% WoW, latency unchanged — bug vs shift vs degradation? **A**: (1) check data pipeline first (schema violations, null spikes, raw-data distribution changes); (2) compare serving vs training feature distributions (Evidently) — if they match, not a pipeline bug; (3) check covariate shift (new transaction patterns / fraud vectors); (4) check label quality; (5) if data clean but performance dropped → model degradation via concept drift.
- **Q4** Three shift types with non-COVID-search examples. **A**: **Covariate** — pre-2020 credit model meets post-COVID irregular incomes; **Label** — diagnosis model moved to a hospital with different disease prevalence; **Concept drift** — 2019 housing price model in the 2021–22 market (same features now predict much higher prices).

**Exercises**
- **Ex 1 — Design a model evaluation gate** (RecSys, Staging → Production): (a) eval dataset and why; (b) metrics; (c) promotion rule (beat prod by X% on Y); (d) gate-failure behaviour. Consider offline NDCG@20 and risk of overfitting to the eval set. (note: lecture template — fresh/rolling held-out set, NDCG@20 ≥ +1%, no critical-slice regression, p99 ≤ 100 ms, diversity drop ≤ 5%; on fail: block, keep prod model, alert, log the run.)
- **Ex 2 — Diagnose a degraded sentiment model** (91% → 78% on weekly manual eval after 3 months): what first, what data to pull, bug vs concept drift, retrain vs roll back. (note: follow the Q3 order — pipeline/data-quality first, serving-vs-training distribution comparison, then label/concept checks; roll back if a recent change caused it, retrain if the world changed.)
- **Ex 3 — Instrument an existing pipeline with MLflow**: log hyperparameters at start, train/val loss each epoch, final test metrics, the model as a registered artefact, tag with git commit; run two experiments and compare in the UI.
- **Challenge — Full MLOps pipeline for ride-sharing demand forecasting** (hourly demand per zone, retrained weekly): specify (1) what you track (data version ID, metrics); (2) registry lifecycle + promotion criteria; (3) CI/CD from commit to production with all test stages; (4) what you monitor and at what cadence; (5) the three retraining triggers with specific thresholds; (6) what constitutes a rollback event and how it's executed; plus a diagram.

**Further reading**: Chip Huyen Ch. 6 (experiment tracking, reproducibility, offline eval) & Ch. 8 (distribution shift & monitoring); MLflow docs (Tracking, Model Registry, Projects; ~20-min quickstart); Evidently AI monitoring guide; Martin Fowler "Continuous Delivery for ML" (CD4ML) — canonical reference for deployment pipeline, eval gates, feature pipelines.

### Likely doubts & exam angles
1. **What's the minimum to make a run reproducible?** Data version (DVC tag/hash), git commit, all hyperparameters, seeds (incl. PYTHONHASHSEED), pinned library versions/Docker image, deterministic GPU ops.
2. **Why isn't S3 enough?** No versioning, lineage, lifecycle, rollback, or audit trail; "which one is live, who deployed it, when?" can't be answered.
3. **Registry lifecycle?** Training → (auto-register) Staging → (passes gate: must beat prod) Production → (new version) Archived; rollback = transition Archived version back to Production.
4. **Is a +0.0004 AUC gain enough to promote?** No — within noise; check slices, latency SLA (p99 ≤ 100 ms), eval-set freshness/overfitting.
5. **What does ML CI add over software CI?** Data validation, model evaluation (perf gate), training regression; and canary (10%) before full rollout.
6. **Covariate vs label vs concept drift?** P(X) vs P(Y) vs P(Y|X); concept drift = same input, different correct output (Wuhan 2019 vs 2020).
7. **First step on a drift alert?** Assume a pipeline bug first (~80% of alerts): null rates, value ranges, timestamp/window errors, ETL changes, scaler stats, training-serving skew.
8. **PSI thresholds?** <0.10 stable; 0.10–0.25 investigate; >0.25 major shift → retrain. KS: p < 0.05 significant.
9. **Which retraining trigger for which domain?** News rec → drift; fraud → performance; medical imaging → scheduled quarterly; credit → scheduled + regulatory. Trade-off: performance-triggered is most trustworthy but slowest and needs labels.
10. **Labels are delayed — what do you monitor?** Proxy metrics: feature drift (PSI/KS/JS), prediction-distribution drift, data-quality signals; don't wait for Layer 4 business metrics.

---

## Cross-links
- **Week 2 → 3.2**: training-serving skew (Week 2) is explicitly called "one of the root causes" of apparent drift; the demo's sklearn `Pipeline` logging (scaler + model as one artefact) is the fix pattern.
- **3.1 ↔ 3.2**: LLM serving decisions (quantization level, self-host vs API) should flow through the same registry/perf-gate discipline — e.g., Quiz Q2 (INT4 rollout) demands task-specific benchmarks + A/B with humans before promotion, exactly like the ML perf gate + canary.
- **Earlier weeks (recsys/ranking)**: promotion rules use **NDCG@20**, slice checks, diversity score, p99 latency ≤ 100 ms — the same offline/online metric vocabulary used in system-design case studies.
- **Later LLMOps weeks** (note: inferred from course title): KV cache, prefix caching (shared system prompts), continuous batching and cost levers (fewer tokens, cheaper model, self-host) underpin RAG/agent cost and latency design; drift monitoring and eval gates carry over to LLM evaluation.

## Glossary
- **Autoregressive generation** — each token is generated conditioned on all previous tokens; strictly sequential.
- **HBM** — high-bandwidth GPU memory; its bandwidth (A100 ≈ 2 TB/s) bounds LLM decode speed.
- **Memory-bandwidth bound** — performance limited by data movement rather than FLOPs.
- **Roofline model** — framework classifying workloads as compute-bound vs memory-bound.
- **KV cache** — stored attention Keys/Values for past tokens; size 2 × L × H × d × S × bytes.
- **Memory fragmentation** — wasted KV memory from pre-allocating max-length contiguous buffers.
- **PagedAttention** — vLLM's paging of KV cache into fixed blocks allocated on demand; shares prefixes/beams.
- **vLLM** — UC Berkeley LLM serving library (PagedAttention, continuous batching, OpenAI-compatible server).
- **TGI / SGLang** — alternative production LLM serving frameworks.
- **Prefix caching** — reusing KV blocks for shared prompt prefixes (system prompts), copy-on-write.
- **Tensor parallel** — splitting a model's tensors across GPUs.
- **FlashAttention / fused kernels** — attention/ops implementations that reduce memory traffic.
- **fp32 / fp16 / bf16** — 4/2/2-byte floats; bf16 has wider exponent range (training stability).
- **INT8 / INT4 / INT2** — 1/0.5/0.25-byte integer quantization.
- **PTQ** — post-training quantization (no retraining).
- **GPTQ** — Hessian-based layer-wise PTQ that compensates rounding error in remaining weights.
- **AWQ** — activation-aware quantization that protects the top ~1% salient weights.
- **Perplexity (PPL)** — LM quality metric; lower is better.
- **GGUF (formerly GGML)** — single-file quantized model format for llama.cpp (Q2_K…Q8_0, Q4_K_M).
- **llama.cpp** — C/C++ CPU-optimized LLM inference (AVX2, ARM NEON).
- **Static batching** — fixed batch run to completion; waits for longest sequence (20–30% util).
- **Continuous (iterative) batching** — refills batch slots after every decode step (70–85% util).
- **Speculative decoding** — draft model proposes K tokens; large model verifies in one pass; 2–3×.
- **Teacher forcing** — feeding known tokens as inputs so all positions are computed in parallel.
- **ONNX / ONNX Runtime** — portable NN IR / Microsoft's optimized runtime (fusion, constant folding).
- **TensorRT** — NVIDIA compiler: layer fusion, INT8 calibration, kernel auto-tuning (2–4×).
- **Break-even (API vs self-host)** — ~$10K/month API spend; >$50K almost always self-host.
- **Experiment / Run / Parameters / Metrics / Artefacts / Tags** — MLflow tracking primitives.
- **DVC** — data version control; log its tag with each run.
- **Model registry** — versioned store with lineage, lifecycle stages, audit trail, rollback.
- **Staging / Production / Archived** — registry lifecycle stages.
- **Promotion gate (perf gate)** — automated rule that new model must beat production before deploy.
- **Canary deployment** — releasing to a small traffic share (10%) before full production.
- **Great Expectations** — data-validation framework (expectation suites: schema, nulls, ranges).
- **Covariate shift** — P(X) changes, P(Y|X) same.
- **Label shift** — P(Y) changes, P(X|Y) same.
- **Concept drift** — P(Y|X) changes; same input, different correct output.
- **Training-serving skew** — features computed differently in training vs serving.
- **KS test** — two-sample Kolmogorov–Smirnov test for distribution difference (p < 0.05).
- **PSI** — Population Stability Index; <0.10 stable, 0.10–0.25 investigate, >0.25 retrain.
- **Jensen-Shannon / KL divergence** — distribution-distance measures for drift checks.
- **Prediction drift** — change in the model's output distribution.
- **Proxy metrics** — label-free signals (feature/prediction drift) used when ground truth is delayed.
- **Scheduled / drift-triggered / performance-triggered retraining** — the three retraining trigger types.
- **Evidently AI, WhyLabs, Arize** — ML monitoring tools.
- **Airflow, Kubeflow Pipelines, Prefect** — pipeline orchestrators.
