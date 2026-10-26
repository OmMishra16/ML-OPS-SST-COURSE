# Course Notes Index — ML System Design and LLMOps

Instructor: Priyansh Saxena (Scaler School of Technology). 8 weeks; course closed Sept 2026.
Course website (shared by the instructor): https://www.priyanshsaxena.com/sst-ml-system-design
Originals: `/Users/ommishra/Desktop/ML/class_content/weekN/` (one folder per week; OA1 paper in `assessments/`; named `wXsY_*`: slides, pre/post handouts, demo notebooks, worksheets, solution designs, case studies).

**Step 0 for every query:** classify it with `exam_guide.md` (concept doubt / descriptive / single-MCQ / multi-MCQ / match / case study) and answer in that category's format. The exam guide also records the instructor's evaluation rules (End-Term 45 marks, Quiz 35 marks).

**How Claude should use this:** when the student asks a doubt, find the week/session below, open only that `weekN.md` (use the line numbers), and answer in the course's own framing, terminology and numbers. Every file follows the same layout: TL;DR → Concepts in lecture order → Demo notebook → Handout extras → Likely doubts → Cross-links → Glossary. Text marked `(note: ...)` is Claude's own addition or correction, not course material. Open the original PDF only if the notes lack the detail.

## Map of the course

| Week | Session | Topic | File:line |
|---|---|---|---|
| 1 | 1.1 | ML Systems 101: notebook→production gap, the 6 components, failure modes (training-serving skew, silent drift, feedback loops), hidden technical debt | week1.md:14 |
| 1 | 1.2 | The ML System Design Framework (how to approach any design question) | week1.md:349 |
| 2 | 2.1 | Feature stores, data pipelines (batch/stream), offline vs online serving, point-in-time correctness | week2.md:12 |
| 2 | 2.2 | Model serving: REST vs gRPC vs streaming, batching, shadow/canary/blue-green | week2.md:300 |
| 3 | 3.1 | LLM serving: memory/bandwidth bounds, KV cache, PagedAttention/vLLM, quantization (GPTQ/AWQ/GGUF), continuous batching, speculative decoding, API vs self-host | week3.md:14 |
| 3 | 3.2 | MLOps: MLflow tracking, model registry, ML CI/CD, drift types, monitoring layers, retraining triggers | week3.md:376 |
| 4 | 4.1 | RAG pipelines & vector DBs: chunking, embeddings, ANN indexes, hybrid search (RRF), reranking | week4.md:13 |
| 4 | 4.2 | Agents & tool use: ReAct, plan-and-execute, failure modes, cost control, Langfuse tracing | week4.md:271 |
| 4 | 4.3 | (Bonus) Four engineering layers of agents: prompt, context, loop, harness | week4.md:513 |
| 5 | 5.1 | The 10-step ML System Design Rubric + real-time music recommender case | week5.md:9 |
| 5 | 5.2 | Spotify Discover Weekly vs Netflix ML pipelines | week5.md:369 |
| 6 | 6.1 | Workshop: LLM-powered semantic/hybrid search at scale | week6.md:9 |
| 6 | 6.2 | How LLM APIs are served at scale (OpenAI, Cohere) | week6.md:235 |
| 7 | — | The 10-step rubric (quick reference) | week7.md:10 |
| 7 | 7.1 | Workshop: End-to-end enterprise RAG | week7.md:37 |
| 7 | 7.2 | LLM monitoring, drift detection and evaluation in production | week7.md:274 |
| 8 | 8.1 | Workshop: Multi-model agentic AI customer support | week8.md:11 |
| 8 | Bonus A | Ride-hailing as an ML system (demand forecasting, upfront fare, surge) | week8.md:247 |
| 8 | Bonus B | Coding agent at scale (N² token cost, prompt-cache economics) | week8.md:709 |
| — | Closing | Course closing | week8.md:990 |
| — | Exam | **Instructor's exam format + how to answer each question type** | exam_guide.md |
| — | Case studies | **Answer format for any case study**: the 10-step rubric exactly as in `w5s1_part1_slides` (+ industry deep-dive format) | case_study_template.md |
| — | OA1 | Online Assessment 1 (W1–W3): questions, answer key and topic tags | OA1_assessment.md |

## Quick topic lookup

| Topic | Where |
|---|---|
| Training-serving skew | week1.md:56 |
| Hidden technical debt (Sculley et al.) | week1.md:179 |
| Feature store | week2.md:182 |
| Point-in-time correctness | week2.md:174 |
| Shadow / canary / blue-green | week2.md:441 |
| KV cache memory math | week3.md:82 |
| PagedAttention | week3.md:104 |
| Quantization | week3.md:128 |
| Continuous batching, speculative decoding | week3.md:201 |
| MLflow / registry | week3.md:411 |
| Drift (classic ML) | week1.md:65, week3.md (S3.2) |
| Drift (LLM systems) | week7.md:405 |
| Chunking | week4.md:60 |
| Hybrid search / RRF | week4.md:135 |
| Reranking | week4.md:147 |
| ReAct vs plan-and-execute | week4.md:314 |
| Langfuse tracing | week4.md:395 |
| Context engineering | week4.md:568 |
| Agent harness (7 pieces) | week4.md:630 |
| Cold start (recsys) | week5.md:309 |
| Prompt caching | week6.md:306 |
| Surge pricing | week8.md:568 |

## Known inconsistencies in the source material (flagged inside the notes)
- W1: one slide says the course runs 12 weeks, but it actually ran 8.
- W3: the slide's GPT-3 KV-cache figure (~100 GB) doesn't match its formula (~9.7 GB per 2048-token sequence). The break-even number printed in the demo doesn't match the demo's own code.
- W4: the RAG refusal threshold is cosine 0.7 in the slides and 0.5 in the notebook. The 4.2 cost-tracker comments don't match the arithmetic. The 4.3 demo notebook isn't in the folder.
- W6: the W6S1 LLM cost of $300/day should be about $8,640/day at the stated rates.
- W7/W8: the W7S1 figure of $150/day and the W8S1 figure of $0.037/conversation are flagged.
- OA1: Q1(b) should be 2.52 GB, not the key's 2.81 GB, so Q1(c) is about 53 requests, not 47. Q2(b) should be about 14.6 req/s, not 20.4.
