# Week 6 — LLM-powered Semantic Search at Scale + How LLM APIs Are Served (OpenAI, Cohere)

Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week6/`): `w6s1_student_worksheet.pdf`, `w6s1_solution_design.pdf`, `w6s2_case_study.pdf`, `w6s2_post_handout.pdf`

(note: Week 6 has no slide deck, demo notebook or pre-handout. Session 1 is a design workshop (worksheet + reference solution) that applies the Week-5 10-step rubric to LLM-augmented hybrid search. Session 2 is a deep-dive case study on how OpenAI and Cohere serve LLM APIs. The W6S2 intro says "You designed an LLM serving system in W6S1". The W6S1 problem is actually LLM-*augmented search*, so treat W6S2 as the "production reality" for the LLM-serving part of that design, and as the answer to the W5S2 preview question.)

---

## Session 6.1 — System Design Workshop: LLM-powered Semantic Search at Scale

### TL;DR
- Problem: semantic search over **100M documents**, **p99 < 500 ms**, **5,000 QPS**, **LLM-augmented answer summaries**, **hybrid retrieval (dense + BM25)**, **new docs indexed within 1 hour**.
- **Key insight (same shape as W5):** "You cannot run LLM inference over 100M documents per query." Use a two-stage pipeline: **hybrid retrieval (100M → top-100, < 150 ms) → cross-encoder reranker (100 → top-20, < 100 ms) → LLM augmentation (answer + citations, < 250 ms, cached)**.
- **The LLM is the latency and cost bottleneck.** Augment only "answer"-intent queries (~10%), use a small LLM (GPT-4o-mini), **cache by MD5 of the normalised query**, enforce a **hard 300 ms timeout with fallback** to plain ranked results, and run a **safety filter** on the output.
- **Most dangerous silent failure:** an **embedding-model version mismatch** between index time and query time causes silent retrieval collapse (NDCG drops, latency unchanged). Fixes: version-tag every request, run a canary reindex on 1% first, and use an automated Recall@100 gate.
- Context: a **precision-first domain**. "A hallucinated LLM summary erodes trust, potentially more than no answer at all." Guardrails: hallucination < 2%, LLM p99 < 300 ms, cost < $0.01/query.

### Concepts (following the 10-step rubric)

(The W6 solution design repeats the 10-step Framework Reference page from Week 5 verbatim. See week5.md, Part 1.)

#### The problem (worksheet)
> Design a semantic search system for a large content platform (e.g., Stack Overflow, GitHub, enterprise knowledge base) with **100 million documents**.
> Requirements: **sub-500 ms p99 latency** · **5,000 QPS** · **LLM-augmented answer summaries** · **hybrid retrieval (dense + sparse BM25)** · **new documents indexed within 1 hour of ingestion**.

Worksheet hint for clarifying questions: "think: *what is a result? when is LLM needed?*"

#### Step 1 — Clarify the Problem
| Clarifying question | Assumed answer |
|---|---|
| What counts as a "result"? Snippet or full document? | Title + passage snippet + optional LLM summary |
| Is the LLM summary always required, or only when helpful? | **Conditional on query intent**: only for "answer" queries, not "find" queries |
| Content domain and languages? | English-only, developer/technical content |
| Availability SLA? | 99.9% |
| LLM cost budget? | **$0.01 per query maximum** |

#### Steps 2 & 3 — Context and Problem Statement
- **Context:** enterprise and developer search is a **precision-first domain**. A missed relevant doc loses productivity time. "**A hallucinated LLM summary erodes trust, potentially more than no answer at all.**" The search bar is often the primary productivity interface, so downtime or a quality regression has immediate, measurable business impact.
- **Problem statement:** given a natural-language query, retrieve the top-K most relevant documents from a 100M corpus and optionally augment with an LLM-generated answer.
  - **Input:** query string.
  - **Output:** ranked list of documents with metadata, plus an optional LLM answer summary **with citations**.
  - **Constraint:** p99 end-to-end < 500 ms, LLM cost < $0.01/query.

#### Step 4 — Define Requirements
| Requirement | Implication for design |
|---|---|
| **Functional** | |
| Hybrid retrieval (dense + sparse) | Parallel ANN + BM25 with **RRF merge**; two index services |
| LLM-augmented answers | LLM sits in the serving path, so it **must have a timeout + fallback** |
| New documents within 1h | Real-time indexing pipeline (Kafka + streaming ingest) |
| Query rewriting / intent detection | Pre-retrieval NLP stage before hitting the indices |
| **Non-functional** | |
| p99 < 500 ms end-to-end | LLM inference is the bottleneck, so **cache aggressively** |
| 5,000 QPS | Stateless serving pods; horizontal autoscaling |
| 99.9% availability | Serving must not depend on training-pipeline health |
| LLM cost < $0.01/query | **Only augment ~10% of queries**; use smaller LLMs + caching |

(Worksheet Step-4 rows: Latency SLA · Throughput (peak QPS) · Availability · Doc freshness SLA · LLM cost per query.)

#### Step 5 — Estimations
| What | Estimate |
|---|---|
| Dense index | 100M docs × **1536 dims** × 4 bytes ≈ **600 GB** |
| BM25 index | Inverted index ≈ **50 GB** |
| Concurrent requests | 5,000 QPS × 500 ms ≈ **2,500 concurrent** |
| LLM cost | 1,000 tokens/query × $0.0002/1K = **$0.0002/query**. 10% of 5K QPS = **500 augmented/s**, given as **$300/day** |
| Embedding new docs | 100 docs/min × 1536 dims, well within one GPU's capacity |
| Cache hit rate target | **40%** for popular queries, saving **$120/day** on LLM costs |

(note: the arithmetic in the source is internally inconsistent. 500 augmented/s × 86,400 s × $0.0002 ≈ $8,640/day, not $300/day. The $120/day saving is 40% of the stated $300. If asked, quote the source figures, explain the method (tokens × price × augmented QPS × seconds/day), and note the discrepancy. The design point stands either way: conditional augmentation (10%) and caching are what keep per-query cost far under $0.01.)

(Worksheet Step-5 rows: dense vector index size · BM25 index size · concurrent requests · LLM cost per day.)

#### Step 6 — System Architecture: the key insight
> "You cannot run LLM inference over 100M documents per query. The solution is a **two-stage architecture**: fast hybrid retrieval narrows the corpus to top-K candidates (e.g., top 20), then the LLM augments only those results. LLM inference is the latency and cost bottleneck in the serving path. **Cache aggressively by query hash.**"

```
Online (<150 ms)              Online (<100 ms)          Online (<250 ms, cached)
Hybrid Retrieval      →       Reranker          →       LLM Augmentation
100M → top-100                100 → top-20              Answer + Citations
```

##### 6a — Data Collection
- Ingests documents from crawlers, push APIs and connectors (**GitHub, Confluence, Jira, Stack Overflow**).
- **Schema per document:** `doc_id, title, body, author, created_at, updated_at, doc_type, source, tags`.
- **Kafka topic** for real-time ingest of new/updated documents. **Schema validation on ingest** prevents corrupt embeddings.
- **Separate cold (bulk historical) and hot (streaming new) pipelines.**

##### 6b & 6c — Data Engineering & Feature Engineering
- **Chunking:** **512-token chunks with 50% overlap**. "Small chunks ⇒ lost context; large chunks ⇒ noisy retrieval. Overlap preserves sentence continuity."
- **Deduplication:** **MinHash LSH** on document bodies removes near-duplicates before indexing.
- **Dense embeddings:** `text-embedding-3-small` (OpenAI) or self-hosted **E5-large**. **Critical:** the embedding model used at index time and at query time **must be identical**. "Version mismatch causes **silent retrieval collapse**."
- **Sparse index:** **Elasticsearch BM25** over title + body tokens. Handles exact-match keyword queries that dense models miss.
- **Metadata filters:** date range, document type, author, source system. Applied as **pre-filters before ANN search**.

##### 6d & 6e — Preprocessing & Models
- **Query understanding (pre-retrieval):**
  - **Intent classifier**: answer vs. find vs. navigate.
  - **Query rewriter**: expands abbreviations and normalises.
  - **Language detection**.
- **Hybrid retrieval:** **dense ANN (FAISS/Pinecone)** and **BM25 (Elasticsearch)** run **in parallel**. **Reciprocal Rank Fusion (RRF)** merges both lists into a single top-100.
  - (note: RRF score of a doc = Σ over lists of 1/(k + rank), with k ≈ 60 typically. It needs only ranks, not comparable scores, which is why it fuses BM25 and cosine scores cleanly.)
- **Reranking:** a **cross-encoder** (e.g. `ms-marco-MiniLM-L12`) rescores top-100 → top-20. "Cross-encoders read query and document **together**: much higher precision than bi-encoders, but too slow for 100M docs."
- **LLM augmentation:**
  - **GPT-4o-mini** (or equivalent) generates a **3–5 sentence answer citing the top-3 documents**.
  - **Only triggered when the intent classifier returns "answer".**
  - The result is **cached by MD5 of the normalised query**.

##### 6f — Post-processing & Serving
**Latency budget (p99 = 500 ms):**
| # | Stage | Budget |
|---|---|---|
| 1 | Network + load balancer | 10 ms |
| 2 | Query understanding (intent + rewrite) | 20 ms |
| 3 | Parallel ANN + BM25 retrieval | 150 ms |
| 4 | Cross-encoder reranking (top-100) | 100 ms |
| 5 | LLM answer generation | 220 ms (0 ms on cache hit) |
| | **Total** | **500 ms** |

- **Safety filter:** LLM output passes through a **content safety classifier** before serving.
- **Fallback:** if the LLM **times out (> 300 ms)** or returns unsafe content, **serve ranked results only (no summary)**. "**Users should not perceive a failure.**"
- (Worksheet asks you to include the **LLM cache and fallback path** in the diagram.)

#### Step 7 — Evaluation: Offline and Online
| Type | Metric | Why it matters |
|---|---|---|
| Offline | **NDCG@10** | Penalises relevant docs ranked lower; the primary retrieval-quality signal |
| Offline | **MRR** (Mean Reciprocal Rank) | How quickly does the first relevant result appear? |
| Offline | **Recall@100** | Is the answer-bearing doc anywhere in the top 100? **A prerequisite for reranking.** |
| Online | **CTR on results (north star)** | Did the user click a result? 7-day rolling average |
| Online | LLM answer satisfaction | Thumbs up/down on generated summaries |
| Online | Zero-result rate | Fraction of queries with no clicks; signals recall failures |
| Online | Query abandonment rate | User left without interacting; signals all results were irrelevant |

**Guardrails (LLM):**
- LLM p99 latency **< 300 ms**.
- **Hallucination rate < 2%** (spot-checked by human raters or an automated **NLI classifier**).
- LLM cost/query **< $0.01**.

"A system that gives **fast, wrong answers destroys user trust faster than slow, correct ones.**"

(note: Recall@100 matters because the reranker can only reorder what retrieval returned. If the answer isn't in the top 100, no reranker can recover it.)

#### Step 8 — A/B Testing
- **Unit:** **user session (search session ID)**. Query-level randomisation risks **within-user contamination**.
- **Treatment options:**
  - (a) LLM-augmented vs. no augmentation
  - (b) cross-encoder reranking vs. none
  - (c) new embedding model vs. current
- **Split:** 10% treatment, then 50/50 after 48h with no guardrail violations.
- **Duration:** **14 days minimum** (weekly patterns; wash out novelty).
- **Bias controls:**
  - **Novelty effect:** users explore more when the interface changes.
  - **Selection bias on query difficulty:** if treatment gets harder queries, CTR may look lower even when quality improved. **Match treatment/control query distributions.**
  - (This is the "bias specific to LLM-augmented search" the worksheet asks for.)
- **Decision rule:**
  - **Promote** if CTR **≥ +1%** AND LLM cost/query **≤ $0.01** AND no guardrail degrades.
  - **Roll back** if **p99 > 600 ms** or **hallucination rate > 2%**.

#### Step 9 — Deployment and Monitoring
**Pipelines:**
- **Airflow** for nightly full-reindex jobs.
- **Kafka consumer** for real-time doc ingest (< 1h SLA).
- **MLflow** tracks embedding-model versions.
- **Automated evaluation gate:** a new embedding model must meet a **Recall@100 threshold on a benchmark** before promotion.

**Model rollout:**
- An **embedding-model upgrade requires a full reindex of 100M docs**. That is expensive, so plan carefully with a **canary reindex on a 1% sample first**.
- **Blue-green for reranker and LLM prompt changes** (no reindex needed).
- **LLM answer cache: Redis, TTL = 24h**. It survives pod restarts.

**6 key metrics to monitor:**
1. **p99 query latency by stage**: retrieval, rerank and LLM separately
2. **Index freshness lag**: minutes since the last document was indexed
3. **Cache hit rate**: target > 40% for LLM augmentation
4. **LLM API error rate**: triggers fallback to non-augmented results
5. **Embedding model version skew**: index and serving versions must match
6. **Zero-result rate**: alert if > 2% of queries return no clicks

#### Step 10 — Risk Management and Trade-offs
| Decision | Choice | Alternative | Why this choice |
|---|---|---|---|
| Retrieval method | **Hybrid (dense + sparse)** | Dense only | Dense is better for semantics; sparse is better for exact keywords (**error codes, function names**). Hybrid outperforms either alone. |
| LLM augmentation | **Conditional (intent = answer)** | Always augment | Always = higher cost + latency on every query. Conditional is cheaper but needs an intent classifier. "Worth the complexity." |
| Embedding model | **Self-hosted E5-large** | OpenAI API | The API is simpler but creates **vendor lock-in and latency variability**. Self-hosted gives control but costs more ops. |
| Chunking | **512 tokens, 50% overlap** | Full doc or small 128-token | Too small = lost context; too large = noisy retrieval. 512 with overlap is "the empirically robust default". |
| Reranking | **Cross-encoder on top-100** | No reranking | Adds 100 ms but **lifts NDCG@10 by ~15%**. Fits within the 500 ms budget. |
| Embedding upgrade | **Canary reindex (1% first)** | Big-bang reindex | A full reindex of 100M docs takes hours. Canary validates quality before committing. |

(Worksheet trade-off rows: dense vs. sparse vs. hybrid · LLM always vs. conditional on intent · self-hosted vs. API embedding model · chunk size small vs. large · reranker yes vs. no.)

#### Failure modes and recovery
| What breaks | How to detect | Recovery |
|---|---|---|
| **Embedding model version mismatch** | Retrieval quality collapses **silently**; NDCG drops, **no latency change** | Compare the index's embedding-model version tag vs. the serving model version **in every request**; **halt serving if mismatched**; trigger a reindex |
| LLM timeout | p99 latency spikes above 500 ms | **Hard timeout at 300 ms**; fall back to non-augmented ranked results; page on-call; degrade gracefully |
| Index staleness (hot content) | Freshness-lag metric rises; users searching for recent docs get no results | Monitor lag per source; **alert if > 90 min**; check **Kafka consumer lag** |
| **Cache poisoning** (bad LLM answer cached) | Thumbs-down rate spikes on a particular query | **TTL-based invalidation (24h max)**; feedback-driven purge via admin API; add an **NLI fact-check layer** |
| BM25 index unavailable | Elasticsearch health check fails; hybrid degrades to dense-only | **Circuit breaker**: fall back to dense-only ANN; alert; **never serve empty results** |

#### Worksheet bonus — "What breaks first when the embedding model is upgraded?"
"Describe the failure mode, how to detect it silently, and how you would safely deploy an embedding-model upgrade." Answer from the reference solution:
- **Failure mode:** if the query encoder is upgraded while the index still holds old-model vectors (or vice versa), query and doc vectors live in **different embedding spaces**. Retrieval quality collapses silently: NDCG/Recall drop while latency and error rate look normal.
- **Detection:** per-request version-tag comparison (index model version vs. serving model version), a dedicated "embedding model version skew" monitor, and offline/online quality metrics (NDCG, zero-result rate, CTR).
- **Safe deploy:**
  1. Promote via MLflow only after passing the **Recall@100 benchmark gate**.
  2. Run a **canary reindex on a 1% sample** and validate quality.
  3. Then do a full reindex (hours for 100M docs), keeping the old index serving until the new one is complete.
  4. Cut over query encoder + index **together**.
  5. Halt serving on any mismatch.
  - (note: building the new index in parallel and swapping atomically is the blue-green pattern applied to indices. Steps 3–4 make explicit what the source implies.)

### Demo notebook
None.

### Pre/Post handout extras
None for W6S1 (worksheet + solution only).

### Likely doubts & exam angles
- **Q: Why hybrid retrieval instead of dense only?** A: dense captures semantics but misses exact tokens (error codes, function names). BM25 catches those. RRF merges both, and hybrid beats either alone.
- **Q: Why RRF?** A: it merges two ranked lists (ANN + BM25) into one top-100 without needing comparable scores.
- **Q: Bi-encoder vs. cross-encoder?** A: a bi-encoder embeds query and doc separately (indexable, fast, used for ANN over 100M). A cross-encoder reads them together (more precise, too slow for 100M), so it is used only on the top-100. It adds ~100 ms for about +15% NDCG@10.
- **Q: How is the LLM kept under budget and SLA?** A: conditional augmentation (only "answer" intent, ~10%), a small model (GPT-4o-mini), a Redis cache by MD5 of the normalised query (TTL 24h, target hit rate > 40%), a hard 300 ms timeout, and fallback to plain results.
- **Q: Why randomise by search session here (vs. user ID in W5)?** A: query-level randomisation causes within-user contamination. The session is the coherent unit of a search task.
- **Q: What is the LLM-specific A/B bias?** A: selection bias on query difficulty (match query distributions) and the novelty effect from the UI change.
- **Q: What are the guardrails?** A: LLM p99 < 300 ms; hallucination < 2% (human raters / NLI); cost < $0.01/query. Rollback triggers: p99 > 600 ms or hallucination > 2%.
- **Q: What is the most dangerous failure?** A: embedding-version mismatch. It is silent (NDCG drops, latency normal). Fix: a version tag in every request, halt on mismatch, reindex.
- **Q: What is cache poisoning?** A: a bad LLM answer cached and served repeatedly. Detect via thumbs-down spikes; mitigate with the 24h TTL, an admin purge, and an NLI fact-check.
- **Q: Chunking choice?** A: 512 tokens, 50% overlap (small = lost context, large = noisy).
- **Q: Dense index size?** A: 100M × 1536 × 4 B ≈ 600 GB (vs. 50 GB for W5's 100M × 128-dim song index). Dimension size matters.
- **Q: Why separate hot/cold ingestion?** A: bulk historical backfill shouldn't block the < 1h streaming SLA for new docs.

---

## Session 6.2 — How LLM APIs Are Served at Scale (Deep-Dive: OpenAI, Cohere, and Anthropic)

### TL;DR
- Core question: "how do you serve a 175-billion-parameter model to millions of users, in real time, at low cost, with multi-tenant isolation?"
- LLM serving is **synchronous, online and autoregressive**. "**Latency is a first-order constraint, not a guardrail**": 50 tokens × 200 ms/token = 10 s.
- **OpenAI:** (1) **stateless serving** (full context in every request, which enables horizontal scaling at the cost of growing input tokens); (2) **model routing** (explicit/implicit, tiered models); (3) **token-bucket rate limiting** on RPM/TPM/TPD (tokens, not requests); (4) **prompt (prefix) caching** of the KV cache (up to 80% cost reduction; exact prefix match only).
- **Cohere:** (1) **LoRA multi-tenant fine-tuning** (one frozen base model plus 10–100 MB adaptors per customer, swapped in ms); (2) **private VPC deployment** for regulated industries; (3) **embedding at scale** (separate GPU embedding cluster + CPU ANN cluster); (4) **multi-region** for latency, data residency and availability.
- "LLM inference is **GPU-memory-bound, not compute-bound**": more users means more GPUs, not faster GPUs.

### Concepts (in case-study order)

#### The unprecedented scaling event (ChatGPT)
- **Nov 30, 2022:** public launch.
- **Dec 4, 2022 (5 days later):** 1M users.
- **Feb 2023 (2 months):** 100M users, the fastest consumer product to 100M in history.
- **Why it's hard:** a single **A100 (80 GB VRAM) serves ~20–30 concurrent GPT-4 requests**, depending on sequence length and quantisation. At **10M requests/day**, you need **hundreds of A100s** running continuously for GPT-4 alone. "Unlike a web server, you cannot simply spin up more instances of a stateless service. **Each inference server must hold the full model in GPU memory.**"

#### The two systems
| | OpenAI API | Cohere Enterprise |
|---|---|---|
| Users | Consumer + developer API | Enterprise B2B customers |
| Scale | 100M+ monthly users | Thousands of enterprise tenants |
| Model tiering | gpt-3.5 / gpt-4 / gpt-4o | Shared + private deployment |
| Key design | Stateless, horizontal scaling | LoRA for per-customer fine-tuning |
| Innovation | Prompt caching, token-bucket rate limiting | Private VPC deployment, embedding at scale |

**The critical difference from recommendation systems:** W5S2 covered batch pipelines (Spotify runs weekly). LLM serving is **synchronous, online and autoregressive**: each token is generated one at a time while the user waits. A recsys with a 200 ms P99 is fine. An LLM generating 50 tokens at 200 ms/token takes **10 seconds**, which is unacceptable for conversation.

---

#### OpenAI API: serving at consumer scale

##### Design 1 — Stateless serving: context lives in the request
"OpenAI's API is **fully stateless**. Every API call includes the full conversation context in the request body. The server has no memory between calls. This is the **foundational design choice that enables horizontal scaling**."
- Every request contains **system prompt + full conversation history + new user message**.
- The server processes it and returns a response **with no write to any database**.
- **The application developer is responsible** for storing and re-sending history.
- The load balancer can route any request to any server, with **no session affinity**.
- **Why it scales:** adding servers adds capacity immediately. The only shared state is the **model weights (read-only, loaded into GPU memory at startup)**. There are no DB hotspots, session locks or distributed-state sync.
- **Trade-off:** context grows with the conversation. Turn 21 re-sends 20 turns, input tokens grow, and **OpenAI charges for input tokens**, so long conversations get expensive. This is "a fundamental tension between **statelessness (scaling simplicity) and cost efficiency**."

##### Design 2 — Model routing: one API, many model versions
- Multiple versions run simultaneously, each on different hardware with different capacity:
  - gpt-3.5-turbo: fast, cheap, small
  - gpt-4: slow, expensive, large
  - gpt-4-turbo: faster GPT-4
  - gpt-4o: multimodal
- **Explicit routing:** the caller specifies `model: "gpt-4"` and the API gateway routes to that cluster.
- **Implicit routing:** products like the ChatGPT web app may route to different versions based on **load, cost or A/B-test assignment**, invisibly to users.

**Capacity planning table**
| Model | Relative cost | GPU memory | Concurrent requests per A100 |
|---|---|---|---|
| gpt-3.5-turbo | 1× | Low | ~100+ (small model, fp16) |
| gpt-4 | 30× | High | ~20–30 (large model, quantised) |
| gpt-4-turbo | 10× | Medium-high | ~40–60 (optimised) |

##### Design 3 — Rate limiting: the token-bucket algorithm
- **Problem:** without limits, one customer's midnight batch job can exhaust GPU capacity and degrade service for everyone. Rate limiting protects **capacity fairness**.
- **Three dimensions:**
  - **RPM**: requests per minute
  - **TPM**: tokens per minute (input + output)
  - **TPD**: tokens per day
- **Token bucket:**
  - The bucket starts full at capacity **C** (e.g. 90,000 tokens).
  - Each request drains it by its token count.
  - It refills at rate **r** (e.g. 90,000 tokens/min = **1,500 tokens/s**).
  - A request that arrives when the bucket is empty is rejected with **HTTP 429 Too Many Requests**.
- **Why tokens, not requests?** Requests vary enormously (10 vs. 10,000 tokens) and use very different GPU resources. Token limits ensure fair **resource** consumption, not just fair request counts.
- **Tiers:** Free (limited), Pay-as-you-go (moderate), Enterprise (custom, negotiated on committed spend).

##### Design 4 — Prompt caching: skipping repeated KV computation
- **Insight:** production apps send the same long system prompt with every request. Without caching, the model **recomputes the KV (key-value) attention cache** for that prompt every time. Prompt caching skips this when the **prefix matches** a cached version.
- **Mechanics:**
  1. Hash the **first N tokens** of the request.
  2. If the hash matches a cached KV tensor in GPU memory, **skip prefill** for those tokens.
  3. Process only the new (uncached) tokens.
  4. Return the response. The cached KV state persists for a **short TTL**.
- **Impact:** **up to 80% cost reduction** for apps with long static system prompts. E.g. a RAG chatbot with a 2,000-token system prompt pays 80% less per query.
- **Constraint:** **exact prefix match only**. A one-character change to the system prompt misses the cache. "Structure prompts with the **static prefix first and the dynamic content (user query) at the end**."
- Flow: `Incoming request (system prompt + history + query) → Prefix hash check (first N tokens hashed) → HIT: skip prefill, save 80% compute | MISS: full prefill, store KV cache`.

---

#### Cohere: enterprise multi-tenant LLM serving

##### Design 1 — Multi-tenant fine-tuning with LoRA
- **Problem:** thousands of enterprise customers each want a model fine-tuned on their proprietary data. "You cannot load 10,000 different fine-tuned models into GPU memory simultaneously." The solution is **LoRA (Low-Rank Adaptation)**.
- **What LoRA is:** instead of fine-tuning all (billions of) weights, inject **small trainable weight matrices (adaptors) into the attention layers**. The full model stays **frozen**. Only the LoRA matrices are trained, typically **a few million parameters** vs. billions. (Paper equation: **W = W₀ + BA**, where B and A are small low-rank matrices; Hu et al. 2021.)
- **Serving architecture:**
  1. **One copy of the base model** (e.g. Command R) in GPU memory: **~30–70 GB** depending on quantisation.
  2. Each customer's LoRA adaptor is stored on disk: **~10–100 MB per customer**.
  3. When a request arrives, the infrastructure loads the right adaptor onto the base model in memory, in **milliseconds**.
  4. Base model + adaptor process the request. The adaptor is applied via low-rank matmul at each attention layer.
  5. If the next request is from a different customer, swap the adaptor.
- **Why it works:** one GPU cluster with one base model serves **thousands of fine-tuned variants** without the memory cost of thousands of full models.
- Diagram: `Base Model (frozen, shared across all customers) + {LoRA A (Customer 1, 10 MB) | LoRA B (Customer 2, 10 MB) | LoRA C (Customer 3, 10 MB)} → Personalised response (per-customer fine-tuned behaviour)`.

##### Design 2 — Private deployment: enterprise data requirements
- **Why:** regulated industries (finance, healthcare, legal) can't send customer data to a third-party API. **GDPR and HIPAA** require data-residency and processing guarantees a shared API can't provide. Enterprises also want their proprietary data to stay out of the shared model.

| Mode | What it means | Who uses it |
|---|---|---|
| Cohere Platform | Shared API on Cohere's infrastructure | Startups, developers, non-regulated enterprises |
| Private Deployment | Model + serving infra run in the **customer's own cloud VPC** | Banks, hospitals, defence, legal firms |

- **Engineering challenge:** package the model + serving infra as a deployable artifact for the customer's AWS/Azure/GCP account. This requires **containerised deployment**, **model-weight distribution** (secure transfer of large files), and **remote monitoring without data egress**.

##### Design 3 — Embedding at scale: GPU + ANN cluster architecture
- **Use case:** enterprise semantic search; 100M docs; sub-100 ms similarity search.
- **Scale problem:** 100M × **1,024 dims** × 4 B ≈ **400 GB** of vectors. That doesn't fit in GPU memory or a single machine's RAM.
- **Architecture:**
  1. **Batch embedding pipeline:** a GPU cluster (A100s) runs Cohere Embed over all docs in parallel. At **~10,000 docs/s per GPU**, 100M docs take **~3 hours on 4 GPUs**. (note: 100M / (4 × 10K/s) = 2,500 s ≈ 42 min of pure compute. The stated ~3 h presumably includes overheads; quote the source figure.)
  2. **ANN index construction:** embeddings are written to FAISS or Pinecone. Index building is a **separate compute job**.
  3. **Online serving:** the query embedding runs on **GPU** (fast, single vector). Similarity search runs on the **ANN cluster (CPU-based, distributed)**. "The two clusters are separate because their **access patterns and scaling requirements differ**."
  4. **Incremental updates:** new docs are embedded and added **asynchronously**. There is **no full rebuild** for small additions.

##### Design 4 — Multi-region deployment
| Concern | Problem | Solution |
|---|---|---|
| User latency | UK user hitting a US API: **+100 ms** network latency | Regional deployments (US-East, EU-West, Asia-Pacific) |
| Data residency | GDPR: EU user data must stay in the EU | EU-West deployment; data never crosses the region boundary |
| Availability | US-East outage affects all users | Route UK traffic to EU-West: adds latency, maintains availability |

#### Key architectural patterns: student design vs. production reality
| Common student design | What OpenAI / Cohere built | The lesson |
|---|---|---|
| Stateful session management (server stores chat history) | Fully stateless; context passed in every request | Statelessness enables horizontal scaling. The cost is larger requests. |
| Single model for all users | Tiered models (3.5 / 4) + model routing | Cost-performance tiering is an **infrastructure** decision, not just a product decision |
| One fine-tuned model per enterprise customer | LoRA adaptors on a shared base model | Fine-tuning for N customers requires **N adaptors, not N model copies** |
| Rate limiting by requests | Token bucket by token count | LLM requests vary 100× in size; **tokens are the right unit** of resource consumption |
| No caching (recompute everything) | Prompt prefix caching (KV-cache reuse) | Up to 80% cost reduction for apps with static system prompts |

(note: the subtitle names Anthropic, but the case-study body covers only OpenAI and Cohere.)

### Demo notebook
None.

### Pre/Post handout extras (w6s2_post_handout)

**Key takeaways (4):**
1. **Stateless serving is the foundation of horizontal scaling.** Any server can handle any request. The trade-off is larger request sizes. "Stateless design is not a simplification; it is a deliberate architectural choice with real costs."
2. **LLM inference is GPU-memory-bound, not compute-bound.** "An A100 GPU can hold approximately one copy of GPT-4 (in quantised form). Serving more concurrent users means **more GPUs, not faster GPUs**." At 10M req/day you need hundreds of A100s. Capacity planning differs fundamentally from web services.
3. **LoRA enables multi-tenant fine-tuning without N model copies.** One base model in GPU memory plus tiny 10–100 MB adaptors loaded dynamically per request. "Fine-tuning at enterprise scale is an **infrastructure problem**, not just an ML problem."
4. **Rate limiting by tokens reflects the true cost driver.** It directly drives OpenAI's pricing model.

**Test Your Understanding, with official answers:**
- **Q1.** A 3,000-token system prompt goes with every call. There are 100,000 calls/day, a 50-token user query, and prompt caching is on. Estimate the input-token savings.
  - **A:** Without caching, each call is 3,000 + 50 = 3,050 tokens, so **305M tokens** in total.
  - With an 80% hit rate: 80,000 × 50 + 20,000 × 3,050 = 4M + 61M = **65M tokens**.
  - **Savings: 240M tokens ≈ 79% of input tokens.** The system prompt is 3,000/3,050 ≈ 98% of input, so caching cuts total input cost by **~79%**.
- **Q2.** 500 enterprise customers each have a LoRA adaptor. Describe the full serving path.
  - **A:**
    1. The HTTP request arrives at the **API gateway** with a customer ID and prompt.
    2. The gateway authenticates and routes to the LLM serving cluster.
    3. The serving system looks up the customer ID to find the right LoRA adaptor.
    4. If the adaptor is not in GPU memory (cold), load it from **object storage** into a memory buffer (ms).
    5. Apply the adaptor to the base model's **attention weight projections** in place.
    6. Run inference: tokenise, forward pass with the modified attention weights, generate tokens autoregressively.
    7. Return the **streamed** response.
  - The base weights sit **permanently in GPU VRAM**. Adaptors live in a smaller memory region and are swapped per request, or **batched** when several requests from the same customer arrive together.
- **Q3.** The TPM limit is 90,000 and requests average 1,500 tokens. What is the effective RPM, and what if requests average 3,000 tokens?
  - **A:** 90,000/1,500 = **60 RPM**. At 3,000 tokens it becomes **30 RPM**.
  - **TPM binds when requests are large; RPM binds when requests are small.** Long-context developers are constrained more by TPM.
- **Q4.** Why does a UK user hitting US-East add ~100 ms?
  - **A:** Light in optical fibre travels at **~200,000 km/s**. London–New York is ≈ 5,570 km, so one way ≈ **28 ms** and the round trip ≈ **56 ms**. Routing overhead, TCP handshakes and data-centre processing bring it to ~100 ms.
  - **Fix: regional deployment.** An EU-West endpoint lets UK users connect in < 10 ms.

**Design exercises (no official answers):**
1. **Cost model for a RAG app:** 2,000-token system prompt; 500-token retrieved chunk; 50-token query; 10,000 conversations/day × 5 turns. Compute (a) input tokens/day without caching, (b) with caching (80% hit on the system prompt), (c) the % of input cost eliminated, using gpt-4o-mini pricing.
   - (note, a worked sketch assuming 1 chunk per turn and ignoring history growth: 50,000 turns × 2,550 = 127.5M tokens/day. With caching: 0.8 × 50,000 × 550 + 0.2 × 50,000 × 2,550 = 22M + 25.5M = 47.5M, i.e. ~63% eliminated. With stateless history re-sending, real totals are higher.)
2. **Design a LoRA serving system for 1,000 small businesses:**
   - (a) adaptor storage/retrieval
   - (b) the latency cost of loading an adaptor per request
   - (c) adaptor already loaded vs. not
   - (d) diagram: be concrete about GPU memory vs. CPU memory vs. disk/object storage
   - (note: by Q2's pattern, the base model lives in VRAM, hot adaptors in a GPU LRU cache, warm adaptors in CPU RAM, and everything in object storage. Batch requests by adaptor.)
3. **Stateless vs. stateful trade-off:** (a) three advantages of a stateful server (UX, cost: e.g. no re-sending of history, fewer input tokens, server-side KV reuse); (b) three disadvantages (scaling via session affinity, failure recovery when sessions are lost on node failure, privacy from storing conversations). Choose a design for **1M DAU** and justify it.

**Challenge, KV-cache and context-window economics:**
- The KV cache grows **linearly with sequence length**, and attention compute grows **quadratically**.
- (a) A 128K-context GPT-4 request uses 40 GB of KV cache on an 80 GB A100. How many concurrent requests fit?
- (b) Compare with a 4K request (~1.25 GB). What does that imply for long-context pricing?
- (c) Propose a KV-cache eviction strategy for mixed 4K/128K traffic.
- (note: (a) 80/40 = **2** at most, before counting model weights, so effectively ≤ 1 in practice. (b) 80/1.25 = **64**, so one 128K request displaces ~32 short ones, which justifies a steep long-context premium. (c) Options: separate pools/queues for long vs. short requests; LRU/priority eviction preferring short-lived short contexts; paged KV memory (vLLM PagedAttention) to avoid fragmentation; preempt-and-recompute or offload long contexts to CPU.)

**Further reading:**
- OpenAI Platform Docs: Rate Limits & Prompt Caching
- **Hu et al. (2021), LoRA** (ICLR 2022; key equation W = W₀ + BA)
- Andrej Karpathy, "Intro to LLMs" (KV cache; why serving is GPU-memory-bound)
- **vLLM / PagedAttention, Kwon et al., SOSP 2023**

**Online resources:** OpenAI Cookbook, vLLM docs (docs.vllm.ai), Cohere docs (LoRA fine-tuning and deployment), LLM Inference Survey (arXiv 2023), Simon Willison's blog, Lilian Weng's blog.

### Likely doubts & exam angles
- **Q: Why is OpenAI's API stateless, and what does that cost?** A: any server can take any request (no session affinity, no DB writes, weights read-only), so scaling is just adding servers. Cost: history is re-sent every turn, so input tokens (and the bill) grow with conversation length.
- **Q: Why can't you scale LLMs like web servers?** A: every replica must hold the full model in GPU memory. Concurrency is limited by VRAM (an A100 serves ~20–30 GPT-4 requests), so more users means more GPUs.
- **Q: Why is latency "first-order" for LLMs but a "guardrail" for recsys?** A: autoregressive generation multiplies per-token latency by output length (50 × 200 ms = 10 s).
- **Q: How does the token bucket work, and what are the limits?** A: capacity C, refill r, drain by tokens, 429 when empty. Limits are RPM/TPM/TPD. Effective RPM = TPM / avg tokens per request.
- **Q: What does prompt caching cache, exactly?** A: the KV attention tensors of a matching prefix (hash of the first N tokens). It skips prefill for those tokens, needs an exact prefix match, and has a short TTL. Put static content first.
- **Q: How does prompt caching differ from W6S1's LLM answer cache?** A: W6S1 caches the *final answer* keyed by the MD5 of the normalised query (Redis, 24h TTL). Prompt caching reuses the *KV state of a shared prefix*, but the model still generates a fresh answer.
- **Q: Why LoRA for multi-tenancy?** A: 1 frozen base (30–70 GB) + N adaptors (10–100 MB), swapped in ms, vs. N full models. W = W₀ + BA.
- **Q: When should you choose private VPC deployment?** A: regulated data (GDPR/HIPAA, finance, health, legal, defence). Challenges: containerisation, secure weight transfer, monitoring without data egress.
- **Q: Why separate the embedding GPU cluster from the ANN CPU cluster?** A: different access patterns and scaling. 400 GB of vectors (100M × 1024 × 4 B) doesn't fit in one machine, so the ANN index is distributed.
- **Q: Why multi-region?** A: latency (UK→US +100 ms, physics of fibre), GDPR residency, and failover availability.
- **Q: What are the model-routing types?** A: explicit (caller names the model) vs. implicit (product routes by load/cost/A-B test). Tiering by cost: 3.5 = 1×, 4-turbo = 10×, 4 = 30×.

---

## Cross-links
- **Week 5:** the same 10-step rubric (the Framework Reference is reprinted in w6s1_solution_design). "Scale forces a two-stage architecture" recurs: W5 is candidate gen → ranker; W6S1 is hybrid retrieval → cross-encoder reranker → LLM. W5 A/B randomises by user ID; W6 by search session. The W5S2 preview question (stateful vs. stateless, pre-computable vs. not, latency target) is answered by W6S2.
- **Week 3 (LLM serving, W3S1; MLOps, W3S2):** the KV cache, prefill, PagedAttention/vLLM, quantisation and batching underlie W6S2's GPU-memory-bound argument. MLflow gates, blue-green/canary and monitoring come from W3S2.
- **Week 4 (RAG, W4S1):** chunking (512 tokens / 50% overlap), embeddings, hybrid BM25 + dense, RRF, cross-encoder reranking, citations and hallucination checks are RAG pipeline components applied at 100M-doc scale. The handout's RAG cost-model exercise links prompt caching to RAG.
- **Week 2 (model serving):** latency budgets, stateless pods, horizontal autoscaling, circuit breakers and fallbacks.
- **Weeks 7–8:** later case-study weeks reuse this design-then-reveal format.

## Glossary
- **Semantic search:** retrieval by meaning using dense embeddings rather than exact keywords.
- **Hybrid retrieval:** dense ANN + sparse BM25 run in parallel and merged.
- **BM25:** sparse lexical ranking function over an inverted index (Elasticsearch). Good at exact terms such as error codes and function names.
- **RRF (Reciprocal Rank Fusion):** merges ranked lists by summing 1/(k + rank). Used to fuse ANN and BM25 into the top-100.
- **Bi-encoder vs. cross-encoder:** separate query/doc embeddings (indexable, fast) vs. joint query+doc scoring (precise, slow; reranks only the top-100).
- **ms-marco-MiniLM-L12:** example cross-encoder reranker model.
- **E5-large / text-embedding-3-small:** self-hosted vs. OpenAI-API embedding models (1536 dims assumed).
- **Chunking (512 tokens, 50% overlap):** splitting docs for embedding. Overlap keeps sentence continuity.
- **MinHash LSH:** locality-sensitive hashing to find and remove near-duplicate documents.
- **Intent classifier (answer / find / navigate):** decides whether to call the LLM.
- **Query rewriter:** expands abbreviations and normalises queries before retrieval.
- **Conditional augmentation:** invoke the LLM only for "answer" intent (~10% of queries).
- **LLM answer cache:** Redis cache keyed by MD5 of the normalised query, TTL 24h, target hit rate > 40%.
- **Cache poisoning:** a bad LLM answer cached and served repeatedly.
- **Hard timeout + fallback:** abort the LLM at 300 ms and serve ranked results only.
- **Safety filter:** content-safety classifier applied to LLM output before serving.
- **NLI classifier:** natural-language-inference model used to detect hallucinations / fact-check answers against sources.
- **MRR:** mean reciprocal rank of the first relevant result.
- **Recall@100:** whether the relevant doc is in the top 100. Bounds what a reranker can achieve.
- **Zero-result rate / query abandonment:** queries with no clicks / users leaving without interacting.
- **Embedding version skew:** index built with a different embedding model than the query encoder. Causes silent retrieval collapse.
- **Canary reindex:** reindex a 1% sample with a new embedding model and validate before a full reindex.
- **Circuit breaker:** automatically bypasses a failing dependency (e.g. BM25 down → dense-only).
- **Kafka consumer lag:** backlog of unprocessed ingest messages; drives index freshness lag.
- **Stateless serving:** the server keeps no session state; the full context is sent in each request.
- **Session affinity:** routing a user's requests to the same server (not needed when stateless).
- **Autoregressive generation:** producing one token at a time, each conditioned on the previous ones.
- **Model routing (explicit / implicit):** the caller picks the model vs. the system picks by load/cost/A-B test.
- **Model tiering:** offering cheap/fast and expensive/strong models (gpt-3.5 1×, gpt-4-turbo 10×, gpt-4 30×).
- **Token bucket:** rate limiter with capacity C and refill rate r, drained by token count. Returns HTTP 429 when empty.
- **RPM / TPM / TPD:** requests per minute / tokens per minute / tokens per day limits.
- **KV cache:** stored attention key/value tensors of processed tokens. Grows linearly with sequence length.
- **Prefill:** the forward pass over the input prompt that builds the KV cache before decoding.
- **Prompt (prefix) caching:** reusing the KV cache for an exact-match prompt prefix. Up to 80% cost reduction.
- **LoRA:** Low-Rank Adaptation. Frozen base weights plus small trainable matrices, W = W₀ + BA. Adaptors are 10–100 MB.
- **Multi-tenant serving:** one base model serving many customers via per-tenant adaptors.
- **Private (VPC) deployment:** the vendor's model and serving stack run inside the customer's cloud account.
- **Data residency:** a legal requirement (GDPR, HIPAA) that data stay within a region or jurisdiction.
- **Multi-region deployment:** regional endpoints for latency, residency and failover.
- **GPU-memory-bound:** serving capacity is limited by VRAM (weights + KV cache), not FLOPs.
- **PagedAttention (vLLM):** paged KV-cache memory management for high-throughput LLM serving.
