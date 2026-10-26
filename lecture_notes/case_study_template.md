# Case Study Answer Format — 10-Step Rubric (w5s1_part1_slides)

**Authoritative source:** `/Users/ommishra/Desktop/ML/class_content/week5/w5s1_part1_slides.pdf`, "ML System Design Rubric, Part 1 of 2: The 10-Step Framework" (12 slides). The instructor's exam guide says: *"You are free to structure your answers as you prefer. However, I recommend following the rubric discussed in class ([w5s1_part1_slides]) for a concise and holistic design."*

How this file is organised:
- **Part 1 = the slide, step by step.** Step names, order, checklists, formulas and quotes are taken from the slide. This is what to write in the exam.
- Lines tagged **[+ from Wk 6–8 solutions]** show how the instructor's own reference solutions (`w6s1/w7s1/w8s1_solution_design`) put each step into practice. They're optional enrichment, not part of the slide.
- Part 2 = a compact exam skeleton. Part 3 = the industry deep-dive format (W5S2/W6S2/W7S2), used only when a question asks how a company built something.

Worked example of the rubric in class: `w5s1_part2_slides`, real-time music recommender, in `week5.md`. More examples: `week6.md:9` (search), `week7.md:37` (enterprise RAG), `week8.md:11` (agentic support).

---

## Why a rubric? (slide 1)
| What interviewers actually assess | What they do **not** expect |
|---|---|
| **Structure**: do you cover all the right dimensions? | The most complex model |
| **Completeness**: do you miss obvious concerns? | Perfect numbers (rough estimates are fine) |
| **Trade-off reasoning**: can you justify your choices? | Knowledge of every tool by name |

**The 10 steps (slide 12, at a glance):**
1. Clarify
2. Context
3. Problem Statement
4. Requirements
5. Estimations
6. Architecture
7. Evaluation
8. A/B Testing
9. Deploy & Monitor
10. Risks & Trade-offs

- **Steps 1–5: Frame the problem.** "Most candidates rush these. Don't."
- **Step 6: Design the system.** "Most time spent here."
- **Steps 7–10:** "Where senior candidates stand out."

---

## Part 1 — The steps

### Step 1 · Clarify the Problem (slide 2)
*"Before designing anything, ask questions."*
- What must the system achieve?
- What metrics define success?
- What are the constraints? (time, data, legal)
- What is out of scope?

**Common clarifying questions:**
- How many users / items / events?
- What latency SLA is acceptable?
- Real-time or batch?
- Is data available and labelled?
- Privacy / regulatory constraints?
- Cost of a bad prediction?

> **Why this matters:** "The problem brief is deliberately incomplete. Asking good questions shows you think like a product engineer, not just a model builder."

In an exam you can't ask, so **write each question with your assumed answer**, e.g. "Catalogue size? (assume 100M songs)". **[+ from Wk 6–8 solutions]**: this is exactly the "Question (assumption)" style the reference solutions use.

### Step 2 · Context (slide 3)
*"Situate the system in the real world."*
- What business problem does this solve?
- Who are the users and what do they need?
- What happens if the system fails?
- What existing systems does this replace?

> **Key principle:** context shapes every decision: model choice, latency budget, fairness requirements, and what counts as success.
> **Example:** a news recommender and an e-commerce checkout recommender use the same algorithms. Context determines which trade-offs matter.

### Step 3 · Problem Statement (slide 4)
*"Formally define the ML problem."*
- What is the **input**?
- What is the **output**?
- What does the model **predict, rank, or generate**?
- How does the ML output **map to a business action**?

> **Weak:** "Build a recommendation system."
> **Good:** "Given a user's interaction history, rank 1,000 candidate songs by probability of play and return the top 20 to the app home screen."
> **Precision here saves time later.** "A vague problem statement leads to architecture debates 30 minutes in. Nail it here."

Template sentence: *"Given <input>, <predict / rank / generate> <output> and <business action>."*

### Step 4 · Define Requirements (slide 5)
| Functional: what it must do | Non-functional: how it must behave |
|---|---|
| Core capabilities and features | Latency SLA (**p99, not mean**) |
| Expected outputs | Throughput (RPS) |
| User-facing behaviours | Availability (**99.9% = 8.7 h downtime/yr**) |
| *e.g. personalised ranking, cold-start, 24h song freshness* | Scalability (**10× headroom**) |
| | Privacy / compliance (GDPR) |
| | Maintainability |

> **Non-functional requirements drive architecture.** "A 100 ms SLA forces a two-stage design. A 10 s SLA allows a single ranker. Requirements are architectural constraints."

**[+ from Wk 6–8 solutions]:** write the requirements as a table with an **"Implication for design"** column (e.g. "p99 < 500 ms → LLM is the bottleneck, cache aggressively"). For LLM systems, add a **cost ceiling** ($/query or $/conversation) as a non-functional requirement.

### Step 5 · Estimations (slide 6)
*"Size the system before designing it."*
- Users / items / events per day
- Request volume (QPS, **peak vs. average**)
- Data volume per day
- Storage (features, models, logs)
- Compute (training hours, serving pods)

| What | Formula |
|---|---|
| QPS | DAU × sessions / 86400 |
| Storage/day | events × bytes/event |
| Feature memory | items × dims × 4B |
| Serving pods | QPS × latency |

> "**Off by 2× is fine. Off by 1000× is a design error.**"
> **Why estimate?** "'Rank 100M songs in real-time' fails the estimation test immediately. No need to build it to know it's wrong."

(note: QPS × latency = concurrent in-flight requests (Little's law). Divide by per-pod concurrency to get pods.)
**[+ from Wk 6–8 solutions]:** for LLM systems also estimate tokens/query, LLM cost/day, and KV-cache/GPU memory (W3S1 formula `2 × L × H × d × S × bytes`). End the step with "→ therefore the design must …".

### Step 6 · System Architecture (slide 7)
**(a) Data Collection → (b) Data Engineering → (c) Feature Engineering → (d) Pre-processing → (e) Models → (f) Post-processing**

| Box | What to cover (slide) |
|---|---|
| (a) Data Collection | sources, signals, ingestion pattern |
| (b) Data Engineering | ETL, storage, schema, quality |
| (c) Feature Engineering | feature store (offline + online) |
| (d) Pre-processing | candidate generation, data prep |
| (e) Models | algorithm, training pipeline, registry |
| (f) Post-processing | re-ranking, filtering, business rules |

> **Design principle:** "For every box: **What does it do? How do you use it? What does it depend on?** If you can't answer all three, the boundary needs work."

**[+ from Wk 6–8 solutions]:**
- open with one **key architectural insight** sentence, e.g. "can't score 100M items in 100 ms → two-stage: candidates then ranking".
- draw a small flow diagram marking **offline vs online** parts and the stage latencies.
- give an explicit **latency budget** (the stage latencies must sum to ≤ the SLA) and a **fallback** for the slow or optional component.
- for LLM and agent problems the box names can be adapted (e.g. W8S1: (a) state management, (d) intent routing, (f) safety & audit). Keep six layers.

### Step 7 · Evaluation: Offline and Online (slide 8)
| Offline (before deployment) | Online (after deployment) |
|---|---|
| Precision@K / Recall@K: relevant items in top-K? | **North star:** CTR, session length |
| NDCG@K: relevant items ranked highest? | **Guardrails:** diversity, error rate, latency |
| AUC-ROC: click-prediction quality | **Good-to-have:** discovery rate, retention |
| *Fast and cheap. Does not measure real behaviour.* | *Ground truth. Slow and expensive to collect.* |

> **The gap between offline and online:** "High NDCG ⇏ high CTR. Offline measures predicted relevance; online measures actual behaviour. The gap is where interesting problems live."

Pick metrics that fit the problem: ranking → NDCG/MRR, classification → AUC/precision/recall, RAG → recall@K and faithfulness, forecasting → WAPE. Always give **one north star, the guardrails, and the good-to-haves**.

### Step 8 · A/B Testing (slide 9)
**What to define:**
- **Unit:** what gets randomised? (user / session)
- **Groups:** treatment (new) vs. control (current)
- **Split:** gradual ramp, 10% → 50% → 100%
- **Duration:** ≥ 2 weeks to capture weekly patterns
- **Sample size:** power analysis. A smaller effect needs a larger sample.

**Biases to flag:**
- **Selection bias:** unbalanced groups
- **Novelty effect:** users engage with anything new
- **Cannibalism:** treatment affects control
- **Simpson's paradox:** aggregate reversal

> **Decision rule:** "Promote if the primary metric improves and no guardrail degrades. Roll back if latency or error rate exceeds its threshold within 24h."

### Step 9 · Deployment and Monitoring (slide 10)
| Deployment | Monitoring: 4 layers |
|---|---|
| Pipelines: Airflow / Kubeflow DAGs | 1. Data quality: schema, nulls, volume |
| Rollout: canary (5% → 100%) or blue-green | 2. Feature drift: vs. training baseline |
| Registry: Staging → Production → Archived | 3. Prediction drift: output distribution |
| Serving API: REST/gRPC; Triton / Vertex AI | 4. Business metrics: CTR, revenue |

> **Alert ordering matters:** "Data quality → feature drift → prediction drift → business impact. Don't wait for the last one to act."

**[+ from Wk 6–8 solutions]:** list "N key metrics to monitor" (6–8), and add the fallback or circuit breaker for each critical dependency. For LLM systems use the W7S2 5-category monitoring framework (operational, input, output, quality evaluation, drift).

### Step 10 · Risk Management and Trade-offs (slide 11)
| Technical trade-offs | Ethical and fairness risks |
|---|---|
| Accuracy vs. latency (bigger model = slower) | **Popularity bias:** rich-get-richer |
| Freshness vs. cost (real-time = expensive) | **Filter bubbles:** users see only what they like |
| Personalisation vs. cold-start | **Feedback loops:** outputs become training data |
| Diversity vs. CTR | **Demographic bias:** protected attributes leak |
| Complexity vs. maintainability | |

> **Interview tip:** proactively raise trade-offs: "*I chose X over Y because Z. The cost of this choice is …*" "Senior candidates don't wait to be asked."

**[+ from Wk 6–8 solutions]:** a compact table, **Decision | Choice | Alternative | Why**, plus **What breaks | How to detect | Recovery** for the top failure modes. Use these only if you have time; the two lists above are the slide's core.

---

## Part 2 — Exam skeleton (concise and holistic)
Cover **all 10 steps, briefly**. Budget the most words for Steps 4–6, but never skip 7–10.
```
1. Clarify      — 4–6 questions, each with "(assume …)"; what's out of scope
2. Context      — business problem · users · cost of failure · what it replaces
3. Problem      — "Given <input>, <predict/rank/generate> <output> → <business action>"
4. Requirements — Functional: … | Non-functional: p99 latency, RPS, 99.9%, 10× headroom, privacy, maintainability
5. Estimations  — QPS = …; storage/day = …; feature memory = …; pods = QPS × latency → "so we need …"
6. Architecture — (a)…(f), one line each (what / how / depends on); key insight; latency budget
7. Evaluation   — Offline: … | Online: north star …, guardrails …, good-to-have …
8. A/B test     — unit, groups, 10→50→100%, ≥2 weeks, power analysis; biases; decision rule
9. Deploy/Mon.  — Airflow/Kubeflow, canary or blue-green, registry stages, REST/gRPC; 4 monitoring layers in alert order
10. Risks       — 2–3 technical trade-offs ("chose X over Y because Z; cost …") + relevant ethical risks
```

---

## Part 3 — Industry deep-dive format (W5S2, W6S2, W7S2)
Only for "how did <company> build X" questions. These follow the session worksheets, not the rubric:
1. **Compare first:** your own design's choices (candidate generation, retrain cadence, cold-start, primary metric, two-stage or not).
2. **The systems side by side:** scale, output, cadence, key tech, cold-start, north star.
3. **Per-company pipeline:** the mechanism step by step, what each model solves that the others can't, the cadence/freshness trade-off and the product assumption behind it, cold-start tiers, metrics.
4. **Compare table** (yours vs A vs B), then **lessons**.
5. Incidents (W7S2): **Symptom → Root cause → Fix → Prevention**.
