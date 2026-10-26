# Week 5 — The ML System Design Rubric + Recommendation Systems Case Study (Spotify & Netflix)

Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week5/`): `w5s1_part1_slides.pdf`, `w5s1_part2_slides.pdf`, `w5s1_student_worksheet.pdf`, `w5s1_solution_design.pdf`, `w5s2_case_study.pdf`, `w5s2_student_worksheet.pdf`, `w5s2_post_handout.pdf`

(note: Weeks 5–8 are case-study weeks. They follow the same format: **Session 1** you design a system from scratch with the 10-step rubric (worksheet + reference solution). **Session 2** compares your design with what real companies built (case study + worksheet + post-handout). Week 5 has no demo notebook and no pre-handout.)

---

## Session 5.1 — The ML System Design Rubric (Part 1) + Real-time Music Recommendation Case Study (Part 2)

### TL;DR
- The **10-step rubric** for any ML system design question: Clarify → Context → Problem Statement → Requirements → Estimations → Architecture (a–f) → Evaluation → A/B Testing → Deployment & Monitoring → Risk & Trade-offs.
- Interviewers grade **structure, completeness and trade-off reasoning**. They do not grade the most complex model, perfect numbers, or knowing every tool by name.
- Steps 1–5 frame the problem ("most candidates rush these, don't"). Step 6 is where most of the time goes. Steps 7–10 are "where senior candidates stand out."
- Case: a music recommender for 50M users, 100M songs, p99 < 100 ms, 10K RPS. **Scale forces architecture**: you cannot rank 100M songs in 100 ms, so you use a **two-stage design**. Offline candidate generation (100M → 1,000) feeds online ranking (1,000 → 20).
- **Budget latency explicitly, not aspirationally.** Use **diversity as a guardrail.** **Cold-start is a product problem**, solved with a tiered fallback. Every component has a failure mode, so design for it upfront.
- Three takeaways: (1) Scale forces architecture. (2) Every component has a failure mode. (3) The rubric works: the same 10 questions apply to any problem.

---

### Part 1 — Concepts (in lecture order)

#### Why a rubric? (slide 1/12)
| What interviewers actually assess | What they do **not** expect |
|---|---|
| **Structure**: do you cover all the right dimensions? | The most complex model |
| **Completeness**: do you miss obvious concerns? | Perfect numbers (rough estimates are fine) |
| **Trade-off reasoning**: can you justify your choices? | Knowledge of every tool by name |

The 10 steps: 1. Clarify the Problem · 2. Context · 3. Problem Statement · 4. Define Requirements · 5. Estimations · 6. System Architecture · 7. Evaluation · 8. A/B Testing · 9. Deployment & Monitoring · 10. Risk & Trade-offs.

#### Step 1 — Clarify the Problem
"Before designing anything, ask questions."
- What must the system achieve? What metrics define success? What are the constraints (time, data, legal)? What is out of scope?
- **Common clarifying questions:** How many users / items / events? What latency SLA is acceptable? Real-time or batch? Is data available and labelled? Privacy / regulatory constraints? What does a bad prediction cost?
- **Why this matters:** "The problem brief is **deliberately incomplete**. Asking good questions shows you think like a **product engineer, not just a model builder**."

#### Step 2 — Context
"Situate the system in the real world."
- What business problem does it solve? Who are the users and what do they need? What happens if the system fails? What existing systems does it replace?
- **Key principle:** context shapes every decision: model choice, latency budget, fairness requirements, and what counts as success.
- **Example:** a news recommender and an e-commerce checkout recommender use the same algorithms. The context decides which trade-offs matter.

#### Step 3 — Problem Statement
"Formally define the ML problem." Cover the input, the output, what the model predicts / ranks / generates, and how the ML output maps to a business action.
- **Weak:** "Build a recommendation system."
- **Good:** "Given a user's interaction history, rank 1,000 candidate songs by probability of play and return the top 20 to the app home screen."
- "**Precision here saves time later.** A vague problem statement leads to architecture debates 30 minutes in. Nail it here."

#### Step 4 — Define Requirements
| Functional: what it must do | Non-functional: how it must behave |
|---|---|
| Core capabilities and features | Latency SLA (**p99, not mean**) |
| Expected outputs | Throughput (RPS) |
| User-facing behaviours | Availability (**99.9% = 8.7 h downtime/yr**) |
| e.g. personalised ranking, cold-start, 24h song freshness | Scalability (**10× headroom**) |
| | Privacy / compliance (GDPR) |
| | Maintainability |

**Non-functional requirements drive architecture.** "A 100 ms SLA forces a two-stage design. A 10 s SLA allows a single ranker. Requirements are architectural constraints."

#### Step 5 — Estimations
"Size the system before designing it." Size users / items / events per day, request volume (QPS, peak vs. average), data volume per day, storage (features, models, logs), and compute (training hours, serving pods).

**Back-of-envelope template**
| What | Formula |
|---|---|
| QPS | DAU × sessions / 86400 |
| Storage/day | events × bytes/event |
| Feature memory | items × dims × 4 B |
| Serving pods | QPS × latency |

- "Off by 2× is fine. **Off by 1000× is a design error.**"
- **Why estimate?** "'Rank 100M songs in real-time' fails the estimation test immediately. You don't need to build it to know it's wrong."
- (note: "QPS × latency" gives the number of concurrent in-flight requests (Little's law). Divide by per-pod concurrency to get pods.)

#### Step 6 — System Architecture (sub-steps a–f)
Flow: **(a) Data Collection → (b) Data Engineering → (c) Feature Engineering → (d) Pre-processing → (e) Models → (f) Post-processing**
- (a) sources, signals, ingestion pattern
- (b) ETL, storage, schema, quality
- (c) feature store (offline + online)
- (d) candidate generation, data prep
- (e) algorithm, training pipeline, registry
- (f) re-ranking, filtering, business rules

**Design principle:** for every box, answer three questions: **What does it do? How do you use it? What does it depend on?** "If you can't answer all three, the boundary needs work."

#### Step 7 — Evaluation: Offline and Online
| Offline (before deployment) | Online (after deployment) |
|---|---|
| Precision@K / Recall@K: are relevant items in the top-K? | **North star:** CTR, session length |
| NDCG@K: are relevant items ranked highest? | **Guardrails:** diversity, error rate, latency |
| AUC-ROC: click-prediction quality | **Good-to-have:** discovery rate, retention |
| *Fast and cheap. Does not measure real behaviour.* | *Ground truth. Slow and expensive to collect.* |

**The offline–online gap:** "High NDCG ⇏ high CTR. Offline measures predicted relevance; online measures actual behaviour. The gap is where interesting problems live."

#### Step 8 — A/B Testing
**What to define:**
- **Unit:** what gets randomised (user / session)?
- **Groups:** treatment (new) vs. control (current)
- **Split:** gradual ramp, 10% → 50% → 100%
- **Duration:** ≥ 2 weeks to capture weekly patterns
- **Sample size:** use power analysis. A smaller effect needs a larger sample.

**Biases to flag:**
- **Selection bias:** unbalanced groups
- **Novelty effect:** users engage with anything new
- **Cannibalism:** treatment affects control
- **Simpson's paradox:** the aggregate result reverses the per-segment result

**Decision rule:** "Promote if the primary metric improves and no guardrail degrades. Roll back if latency or error rate exceeds its threshold within 24h."

#### Step 9 — Deployment and Monitoring
**Deployment:**
- Pipelines: Airflow / Kubeflow DAGs
- Rollout: canary (5% → 100%) or blue-green
- Registry stages: Staging → Production → Archived
- Serving API: REST/gRPC, Triton / Vertex AI

**Monitoring, 4 layers:**
1. Data quality: schema, nulls, volume
2. Feature drift: vs. the training baseline
3. Prediction drift: output distribution
4. Business metrics: CTR, revenue

**Alert ordering matters:** data quality → feature drift → prediction drift → business impact. "Don't wait for the last one to act." (note: earlier layers are leading indicators. Business metrics lag.)

#### Step 10 — Risk Management and Trade-offs
| Technical trade-offs | Ethical and fairness risks |
|---|---|
| Accuracy vs. latency (bigger model = slower) | **Popularity bias:** rich-get-richer |
| Freshness vs. cost (real-time = expensive) | **Filter bubbles:** users see only what they like |
| Personalisation vs. cold-start | **Feedback loops:** outputs become training data |
| Diversity vs. CTR | **Demographic bias:** protected attributes leak |
| Complexity vs. maintainability | |

**Interview tip:** proactively raise trade-offs: "*I chose X over Y because Z. The cost of this choice is …*" "Senior candidates don't wait to be asked."

#### The rubric at a glance (slide 12/12)
- **Steps 1–5: frame the problem.** Most candidates rush these. Don't.
- **Step 6: design the system.** Most of the time is spent here.
- **Steps 7–10:** where senior candidates stand out.

**From the Framework Reference (solution_design p.1), how to use the rubric in an interview:**
- Work through Steps 1–3 **aloud with the interviewer**, who may constrain the problem further.
- Spend the most time on Steps 4–6. This is where architectural thinking is assessed.
- Steps 7–10 differentiate senior candidates.
- "You do not need to cover every sub-bullet, but you must be able to **justify every decision** you make."

The reference also lists the expected content per step. It matches the slides, plus:
- Step 4 functional examples: personalisation, real-time inference, feedback collection.
- Step 8: statistical significance threshold.
- Step 9: a "Model controller" (version management, blue-green/canary, rollback) and dashboards (latency, drift, business metrics).
- Step 10: data bias and fairness, interpretability vs. accuracy, staleness vs. cost, cold-start, failure modes, ethical implications.

---

### Part 2 — Case Study: Real-time Music Recommendation System (rubric applied)

#### The problem brief
> Design a real-time music recommendation system for a streaming service with **50 million active users**.
> - Personalised recommendations
> - **p99 latency < 100 ms**
> - **10,000 requests per second** at peak
> - **New songs visible within 24 hours**
> - **Cold-start handling** for new users

#### Steps 1–3 — Clarify, Context, Problem Statement
**Clarifying Q&A:**
| Question | Answer (assumption) |
|---|---|
| Catalogue size? | **100M songs** |
| Personalisation granularity: per-user or per-user-session? | **Per-user, updated nightly** |
| Availability SLA? | **99.9%** |
| Scope: the app too, or only ML? | **ML system only, not the app** |

**Context:** music streaming competes on engagement and discovery (solution: "competes for daily active listening time"). Personalised recommendations drive session length, retention, artist discovery and subscription retention.

**Problem statement (slide):** "Given a user's interaction history and current session context, rank a candidate set by probability of play and return the top 20." **Input:** user ID + session signals. **Output:** ranked list of 20 song IDs. **Constraint:** ≤ 100 ms end-to-end.

**Problem statement (solution doc, long form):** for each of 50M active users, return a ranked list of 20 songs at the moment they open the app. It must be personalised, meet 100 ms p99, handle 10K RPS peak, expose new songs within 24h of upload, and gracefully serve new users with no history.

#### Step 4 — Requirements
| Requirement | Implication for design |
|---|---|
| **Functional** | |
| Personalised ranking per user | Needs a per-user model or per-user candidate generation |
| New songs indexed within 24h | Candidate generation must update at least daily |
| Cold-start: useful recs with 0 history | Fallback strategy required before interaction history exists |
| Feedback: plays, skips, likes | (log all of them) |
| **Non-functional** | |
| p99 latency < 100 ms | Cannot run full model inference over 100M songs in real time |
| 10,000 RPS at peak | Serving must be **stateless** and horizontally scalable |
| 99.9% availability | Serving layer must **not depend on the training pipeline being healthy** |
| Graceful degradation if cache fails | (fallback path to popularity) |

**Key implication:** "100 ms p99 + 100M songs = you cannot rank all songs at serving time. **This single constraint determines the entire architecture.**"

#### Step 5 — Estimations
| What | Estimate | Implication |
|---|---|---|
| Daily events | ~500M plays/skips × ~500 bytes/event | **≈ 250 GB/day** of event data |
| User features | 50M × 128 dims × 4 B | **≈ 25 GB in Redis** |
| Song ANN index | 100M × 128 dims × 4 B | **≈ 50 GB FAISS index** |
| Serving compute | 10K RPS × 90 ms | **≈ 900 concurrent requests**, so 10–20 GPU serving pods |
| Training compute | 15B training pairs/night (30-day window) | **4–8 GPU hours nightly** |

"What these numbers tell us: a 50 GB ANN index **fits in memory on one machine**. 250 GB/day of events means **nightly batch is sufficient**. Streaming is not needed for the 24h SLA."

#### Step 6 — Architecture: the key insight
> "You cannot rank 100 million songs in real-time under 100 ms. **So you don't.** Pre-filter offline to ~1,000 candidates per user, then rank online in real time."

```
Offline (nightly)           Online (<90 ms)          Online (<10 ms)
Candidate Gen        →      Ranking Model      →     Serving Layer
100M → 1,000                1,000 → 20               Top-20 to user
```
"Every major recommender uses this pattern: Spotify, Netflix, YouTube, TikTok, Amazon."

#### Step 6 — Components
- **6a Data Collection:** Kafka streams of plays, skips, likes, searches (solution adds skip-at-second, like/save, playlist adds). "**Log all events. Skips are signal too.**" What users skip tells you as much as what they play. Schema validation at ingestion (**Great Expectations**) prevents silent data corruption. Song metadata (genre, mood, BPM, artist) lives in a separate offline metadata store.
- **6b–6c Data Engineering & Feature Store:**
  - **Offline store:** S3 + Parquet. Nightly Spark jobs aggregate events into user-song interaction history. Joins are **point-in-time correct** to prevent leakage: features for a training row at date D may only use data available before D.
  - **Online store:** Redis, < 5 ms lookup (solution says "sub-millisecond reads"). Holds the latest user feature vectors and pre-computed candidate lists. **Refreshed every 4 hours.**
  - **Key features.** User: recency-weighted play counts per genre/artist, skip rate, session length. Song: audio embeddings (spectrogram-based), popularity score, age since upload.
- **6d Candidate Generation:** two-tower model → FAISS ANN → 1,000 candidates cached in Redis.
- **6e Models:**
  - **Candidate gen:** two-tower NN. The user tower encodes interaction history. The song tower encodes audio features + metadata. Both are trained jointly into a shared embedding space, and ANN (FAISS) retrieves the 1,000 closest songs. Trained nightly on **implicit feedback (plays)**, not explicit (likes). Plays are **100× more abundant**.
  - **Ranker:** **LightGBM** on user-song feature pairs. Features: user-song affinity score from the two-tower, song popularity, recency, diversity penalty. Runs online over the 1,000 candidates.
  - **Registry:** MLflow versions both models **independently**, so changing the ranker does not require rebuilding the ANN index.
- **6f Post-processing:**
  - Deduplicate artists: no same artist twice in the top 5.
  - Inject novelty: 1 unfamiliar artist per 5 recs.
  - Apply business rules: licensed content only, geo-restrictions.
  - Return the top 20. Serving pods are stateless and horizontally scalable.

#### Step 6 — Latency budget (p99 = 100 ms)
| # | Operation | Budget (ms) |
|---|---|---|
| 1 | Network + load balancer | 5 |
| 2 | Fetch candidate list from Redis | 5 |
| 3 | Fetch user context features from Redis | 5 |
| 4 | Run ranker on 1,000 candidates (**Triton**) | 75 |
| 5 | Serialize + network | 10 |
| | **Total p99 target** | **100** |

"**Budget explicitly, not aspirationally.** Each step has an allowance. If step 4 takes 90 ms, steps 1–3 must fit in 10 ms combined. Design to the budget, then measure it."

#### Step 7 — Evaluation
| Type | Metric | Why it matters |
|---|---|---|
| Offline | Precision@20 / Recall@20 | Fraction of the top-20 the user actually engaged with in the held-out period |
| Offline | NDCG@20 | Penalises relevant songs ranked lower; rewards correct ordering |
| Offline | AUC-ROC | Binary click-prediction quality of the ranker |
| Online | **CTR (north star)**, 7-day rolling | Primary engagement signal |
| Online | Session length | Total time spent after receiving recs |
| Online | Discovery rate | Fraction of recs that are new-to-user artists |
| Online | **Diversity score** | Mean intra-list cosine distance; guardrail against popularity collapse |

**Guardrails:**
- Diversity score must not drop **> 10% week-over-week**.
- Cold-start CTR must be **within 20% of warm-user CTR**.

**Why diversity as a guardrail?** "Recommending the same 100 popular songs to everyone raises CTR short-term but destroys long-term retention. Without a diversity guardrail **the optimiser will exploit this**."

#### Step 8 — A/B Testing
- **Unit:** **User ID**, not session. Session-level randomisation would cause **novelty contamination**. (note: one user would see both variants.)
- **Treatment** = new model version. **Control** = current production model.
- **Split:** 10% treatment / 90% control, then **50/50 after 48h** if no guardrail is violated.
- **Duration:** **14 days minimum**, to capture weekly patterns and wash out the novelty effect.
- **Sample:** 5M users in treatment detects a **1% CTR lift at α = 0.05, power = 0.80**.
- **Bias controls:**
  - Keep a **holdout group (~1% of users) that never sees the new model**, for long-term measurement.
  - Check for selection bias: treatment and control distributions must match on **age, device, listening frequency** before starting.
- **Decision rule:**
  - **Promote** if CTR rises **≥ 0.5%** and diversity does not drop **> 5%**.
  - **Roll back** if p99 **> 120 ms** or error rate rises **> 0.1%** within 24h.

#### Step 9 — Deployment and Monitoring
**Deployment:**
- **Airflow DAGs** run nightly training plus the **4h candidate refresh**.
- **MLflow** tracks every run: data version, code hash, hyperparameters, metrics.
- **Automated evaluation gate:** the new model must **beat production on NDCG@20** before promotion.
- **Blue-green:** the new version serves **10% shadow traffic** first, before the full swap.
- MLflow Registry stages: Staging → Production → Archived.
- **One-command rollback:** demote the current version, promote the previous one.

**6 metrics to monitor:**
1. **p99 latency**: alert if > 120 ms for 5 min
2. **Feature store freshness lag**: hours since the last batch update
3. **Diversity score**: mean intra-list cosine distance
4. **CTR**: 7-day rolling; alert on > 5% week-over-week drop
5. **Cold-start vs. warm CTR gap**: should narrow over the first 5 plays
6. **Candidate list staleness**: hours since the last Redis refresh per user

#### Step 10 — Risk Management and Trade-offs
| Decision | Choice | Alternative | Why |
|---|---|---|---|
| Retrieval | **Two-stage** (cand. gen + ranker) | Single ranker over all items | Can't rank 100M in < 100 ms. Two-stage is the only feasible option at this scale. |
| Candidate freshness | **4h pre-compute** | Real-time ANN at serving time | 4h meets the 24h SLA and keeps latency predictable. Real-time ANN at 10K RPS is prohibitively expensive. |
| Ranker | **LightGBM** | (Small) neural net | GBT trains faster, is more interpretable, and performs comparably on tabular ranking features. |
| Training signal | **Plays (implicit)** | Likes (explicit) | Likes are sparse (**< 2% of plays**). Plays are 100× more abundant: noisier but sufficient. |
| Cold-start | **Tiered fallback** (popularity + survey + fast online learning) | None | A user with nothing to show churns immediately. Popularity is good enough for the first session. |
| Diversity | **Intra-list penalty in ranker** | None | CTR alone optimises toward popularity collapse, and long-term retention degrades. |

**Ethical risk, popularity bias:** without the diversity guardrail, the ranker recommends popular songs aggressively and new artists get no exposure. **CTR goes up, long-term retention goes down.**

#### Cold-start: a product problem, not an ML problem
"Cold-start is a **product problem masquerading as an ML problem**. The fix is a **tiered fallback**, not a better model."

| User state | Strategy | Data source |
|---|---|---|
| 0 interactions | Global popularity + editorial playlists | Aggregated play counts (all users) |
| 1–4 interactions | Content-based: match audio features of played songs | Song metadata/audio embeddings (genre, BPM, mood) |
| 5–19 interactions | Collaborative + content hybrid | Partial history + song embeddings |
| 20+ interactions | Full personalised two-tower retrieval | Full interaction history in the feature store |

"The transition happens **automatically** as interactions accumulate. The feature store drives it, with no manual intervention." (note: the end of this slide's text is garbled. "no manual intervention" is the evident completion.)

#### Failure modes and recovery
| What breaks | How to detect | Recovery |
|---|---|---|
| Redis candidate cache fails | p99 latency spike; cache hit rate drops to 0 | Fall back to global popularity; page on-call; rebuild cache from the offline store |
| Nightly batch training fails | MLflow shows no new run; staleness alert fires | Serve the previous model version; trigger manual retraining |
| Training-serving feature skew | Feature distributions diverge (Evidently); **CTR drops while latency is unchanged** | Pause new-model promotion; compare offline/online distributions; fix the pipeline |
| Ranker model degrades | CTR drops > 5% week-over-week; diversity score falls | Roll back via the registry; investigate training-data quality |
| Cold-start loop | Cold-start CTR ≫ warm CTR; no transition to personalisation | Check interaction logging; verify the feature store is updating for new users |

#### Wrap-up: what we covered
- Part 1: the 10-step framework. Steps 1–5 frame, Step 6 has the (a)–(f) sub-steps, Steps 7–10 validate and operate.
- Part 2: two-stage architecture (the key insight); explicit latency budget; diversity as a guardrail; cold-start as a tiered product solution; failure modes per component.
- **Three takeaways:**
  1. Scale forces architecture: 100 ms + 100M songs = two-stage, always.
  2. Every component has a failure mode. Design for it upfront.
  3. The rubric works: same 10 questions, any problem.
- Closing line: "Every ML system design question has the same ten questions hidden inside it."

### Demo notebook
None for this session.

### Worksheet (w5s1_student_worksheet) — what students fill in
- **Steps 1–3:** clarifying questions; context (business objective, users, cost of failure); problem statement (input, output, constraint).
- **Step 4 table:** Latency SLA · Throughput (peak RPS) · Availability · Freshness SLA · Cold-start, each with value/assumption and design implication.
- **Step 5 table:** daily event volume · user feature store size · song embedding index size · serving pods needed.
- **Step 6:** boxes for (a–c) Data Collection, Engineering & Feature Store; (d–e) Preprocessing / Candidate Gen. & Models; (f) Post-processing plus the full diagram ("labelled boxes, arrows, label every arrow").
- **Step 7:** offline · online north-star · guardrail, plus a "why this metric?" for each.
- **Step 8:** unit, split, duration, sample size, one bias to flag.
- **Step 9:** deployment (pipeline, rollout, registry) plus 6 metrics to monitor.
- **Step 10 trade-off table:** pre-computed vs. real-time candidates; two-stage vs. single ranker; implicit vs. explicit feedback; cold-start strategy; diversity vs. CTR.
- **Bonus:** "**What breaks first at 10× load?** Identify the bottleneck and explain how you would fix it." (note: no official answer is given. Grounded reasoning from the numbers: 10× = 100K RPS ≈ 9,000 concurrent requests. The **ranker on 1,000 candidates** (75 ms budget, GPU/Triton pods) is the obvious first bottleneck, followed by Redis read throughput. Fixes: scale stateless ranker pods horizontally, shrink the candidate set (e.g. 1,000 → 500), use a lighter ranker / batching, Redis read replicas or sharding, and cache the top-20 for repeated opens.)

### Likely doubts & exam angles
- **Q: Why p99 and not mean latency?** A: SLAs are about the tail. The mean hides the slow requests users actually feel. The rubric explicitly says "p99, not mean".
- **Q: What does 99.9% availability mean in downtime?** A: ≈ 8.7 h/year.
- **Q: Why is two-stage mandatory here?** A: 100M songs cannot be scored in 100 ms. Offline candidate gen (100M → 1,000 via two-tower + FAISS ANN) plus an online ranker (1,000 → 20) makes the online work bounded.
- **Q: Why does nightly batch suffice when we have a 24h freshness SLA?** A: the estimates (250 GB/day of events) show batch handles the volume, and nightly + 4h candidate refresh meets 24h. Streaming would add cost without meeting a requirement.
- **Q: Why randomise by user, not session?** A: session-level randomisation exposes the same user to both arms. That causes novelty contamination and inconsistent experience.
- **Q: Why is diversity a guardrail and not the north star?** A: CTR is the north star. Diversity must not degrade (> 10% WoW) because CTR optimisation collapses toward popularity, which hurts long-term retention.
- **Q: Why LightGBM over a neural ranker?** A: faster training, interpretable, comparable accuracy on tabular ranking features.
- **Q: Why version the two models separately in MLflow?** A: so the ranker can change without rebuilding the 50 GB ANN index.
- **Q: How do you detect training-serving skew?** A: feature-distribution drift (Evidently), with the signature "CTR drops, latency unchanged".
- **Q: What is the order of monitoring alerts?** A: data quality → feature drift → prediction drift → business metrics. Act on the leading indicators.
- **Q: Which A/B biases should you flag?** A: selection bias, novelty effect, cannibalism, Simpson's paradox.
- **Q: What is the serving-pods formula?** A: QPS × latency = concurrency (10K × 90 ms ≈ 900), which leads to 10–20 GPU pods.

---

## Session 5.2 — How Spotify and Netflix Build Their ML Pipelines (Deep-Dive Case Study)

### TL;DR
- You compare your W5S1 design with production. "The industry converged on **simpler, not more complex**, architectures."
- **Spotify Discover Weekly** (30M users, every Monday, 30-song playlist) is a **three-model ensemble**: word2vec on playlists ("playlists are sentences, songs are words"), an audio CNN (cold-start for new songs), and NLP on web text (cultural context). It is a weekly batch on Hadoop + Spark.
- **Netflix** (200M+ members) runs a **two-tower candidate gen (retrained daily, ANN rebuilt)** feeding a **separate neural ranker (retrained hourly)** with rich interaction features. The north star is **streaming hours, not CTR**, plus post-play surveys and 30-day retention.
- "The sophistication is not in the model family. It is in **how the pipeline is engineered** around those models": freshness strategy, cold-start handling, and evaluation design.
- Five lessons: proven architectures; freshness is first-class; evaluation determines what is optimised; cold-start is a product problem; scale forces separation of concerns.

### Concepts (in case-study order)

#### The two systems
| | Spotify Discover Weekly | Netflix Recommendations |
|---|---|---|
| Scale | 30 million users, every Monday | 200+ million members |
| Output | 30-song personalised playlist | Ranked content rows |
| Cadence | Weekly batch job | Candidate model **daily**, ranker **hourly** |
| Key tech | word2vec on playlist data | Two-tower neural network + ANN |
| Notable | Cold-start: CNN audio embeddings for new songs | North star: streaming hours, not CTR |

#### Compare your W5S1 design: five dimensions
| Dimension | Spotify | Netflix |
|---|---|---|
| Model architecture | word2vec on playlists + audio CNN | Two-tower neural network |
| Serving strategy | Weekly batch, pre-computed | Candidate daily, ranker hourly |
| Cold-start | Audio CNN for new songs | Feature engineering (recency, genre) |
| Evaluation north star | Implicit: stream completion rates | Streaming hours (not CTR) |
| Scale of freshness | Weekly update cycle | Hourly ranker retraining |

**Key pattern:** both use architectures "you have seen before" (word2vec, neural nets, ANN search). "Freshness strategy, cold-start handling, and evaluation design are where production systems differ from classroom projects."

---

#### Spotify Discover Weekly (published engineering blog; 30M users; weekly batch; three-model ensemble)

##### Model 1 — Collaborative filtering via word2vec on playlists
**Key insight:** "Treat **playlists as sentences**. Treat **songs as words**. Apply word2vec." Songs that co-occur in many playlists end up close in embedding space, regardless of genre label, audio features or explicit tags. "The playlist context is the signal."

Mechanism:
1. Aggregate all user playlists across the platform. Each playlist is a sentence: an ordered sequence of song IDs.
2. Train a **skip-gram word2vec** model. Each song gets a **128-dim** embedding such that songs in similar playlist contexts have high cosine similarity.
3. Example: users A and B both have Radiohead and Thom Yorke. A also has Portishead. So Portishead's embedding lands near Radiohead's, and Portishead is a reasonable recommendation for B.
4. Result: **40+ million song embeddings**. Songs close in this space share **cultural context, not just acoustic similarity**.

Flow: `Playlists (billions of rows) → offline word2vec skip-gram training → song embeddings (128-dim) stored in index → at serving time: ANN search (cosine similarity)`.

(Worksheet answer to "Why does this work without any audio analysis?": because users themselves put songs that belong together into the same playlists. Co-occurrence encodes similarity, the same way word2vec learns word meaning from context.)

##### Model 2 — Content-based filtering via audio CNN
- **What:** every song's raw audio goes through a CNN trained to predict audio features (genre tags, mood, instrumentation). The **final hidden layer** gives an embedding based purely on acoustic content, with no playlist co-occurrence required.
- **Why:** new songs cannot appear in word2vec embeddings because they are in no playlists yet. The audio CNN gives **every new song an embedding from day 1**. This is **Spotify's cold-start solution for new music**.
- **Architecture:** input is a **log-mel spectrogram**. The CNN extracts timbral, rhythmic and structural features. Multiple conv layers with max-pooling reduce it to a fixed-length embedding.

##### Model 3 — NLP on text: cultural-context embeddings
- Crawl the internet for text about artists/songs (lyrics, reviews, blogs, social posts, music journalism) and apply word2vec to that corpus. The embeddings capture **cultural meaning**: the community and discourse around an artist.
- Example: post-rock artists cluster together not because their playlists overlap, but because journalists describe them with the same vocabulary.

**Why three models?** Each embedding space captures a different aspect of similarity: playlist co-occurrence, acoustic content, cultural discourse. Combining them makes recommendations more robust.

##### Ensemble: three embedding spaces into one score
At recommendation time:
1. Compute the user's listening-history embedding as the **average of all song embeddings in their history**.
2. For each candidate song, compute cosine similarity in **all three spaces**.
3. Take a **weighted combination** of the three similarity scores.
4. Rank by the combined score and return the **top 30**.

Timing:
- Weekly batch on **Hadoop + Spark**. Every Monday, all 30M Discover Weekly playlists are regenerated.
- Total time ~**30 minutes** for all users (at Hadoop scale).
- Songs that appear in playlists this week show up in next Monday's embeddings.

##### Spotify cold-start for new songs
Problem: a song released today has zero playlist co-occurrences. **> 40,000 songs are released per day** on Spotify, so "this is not an edge case."

| Song age | Available signal | Embedding source |
|---|---|---|
| 0–7 days | No playlist data | **Audio CNN only** |
| 7–30 days | Limited playlist co-occurrences | Audio CNN + partial collaborative |
| 30+ days | Rich playlist data | Full three-model ensemble |

**Design implication:** Spotify maintains **two separate embedding indices**. One holds the full collaborative signal (updated weekly). The other holds audio-only embeddings (updated as new songs are uploaded). The serving layer **queries both and blends** the results. "A deliberate **separation of concerns** in the pipeline, not an afterthought."

##### Spotify end-to-end pipeline
```
User Playlists (Hadoop HDFS) → Spark Jobs (weekly batch) → word2vec Training (song embeddings) ┐
                                                          Audio CNN (new song embeddings) ─────┤
                                                                                               → User Profile (history avg vector) → Serve Top-30 (every Monday)
```

##### Spotify's freshness trade-off (worksheet question)
"Why weekly, not hourly like Netflix?" (note: the source doesn't print an answer. From the material: Discover Weekly is a *product* built around a weekly ritual, a fixed playlist delivered every Monday. Musical taste changes slowly. Batch at Hadoop scale is cheap (~30 min/week). The embedded product assumption is that **users' music taste is stable over a week, and the product value is a curated weekly drop, not a reaction to in-session intent.**)

---

#### Netflix: two-tower + ranking (published research; 200M+ members; candidate daily, ranker hourly; streaming hours north star)

##### Stage 1 — Candidate generation: two-tower NN
- A **user tower** and an **item tower**, trained jointly on watch events. At inference, **ANN search retrieves the top 1,000** candidates.
- User tower inputs: watch history, ratings, device / time, demographics.
- Item tower inputs: genre / language, recency, popularity, cast / director.
- Shared embedding space with **dot-product similarity**, then ANN index (FAISS), then top-1000 candidates.
- **Training:** implicit feedback. **Positive pair = (user, title watched > 50%)**. Negatives are randomly sampled titles. The loss (softmax or **in-batch negatives**) pulls watched items closer to the user embedding and pushes unwatched ones away.
- **Retrained daily.** The ANN index is rebuilt after each training run.
- (Worksheet Q, "Why can item embeddings be pre-computed offline but not user embeddings?": note, item features change slowly and are shared across all users, so you embed the catalogue once and index it. The user embedding depends on current context (device, time, latest watches), so it is computed per request and used as the ANN query.)

##### Stage 2 — Ranking: a separate NN with rich features
- **Why a separate ranker?** The two-tower model must stay simple enough to **build an ANN index over all items**. The ranker has no such constraint. It runs over only 1,000 candidates and can use **expensive, interaction-level features that can't be pre-computed per item**.
- **The ranker optimises three things together:**
  1. Probability of play (will the user click?)
  2. Expected play duration (will they watch > 75%?)
  3. Post-play survey score (did they rate it positively?)
- **Retrained hourly**, to capture within-day context changes. Example: a user browsing at 11 pm on a Friday has a different intent than at 8 am on a Tuesday.

**Netflix feature engineering**
| Feature category | Examples |
|---|---|
| User history | Watch time per genre, ratings given, completion rates, time since last watch |
| Context | Device type (TV vs. mobile), time of day, day of week, country |
| Item features | Genre, language, recency of release, average rating across all users |
| Interaction (u × i) | Has the user watched other titles by this director? Similar-genre completion rate? |

##### Netflix multi-objective evaluation
**Insight:** "Maximising CTR alone produces **clickbait**." Netflix optimises for **long-term member satisfaction**. A model that recommends a mediocre show the user starts and abandons is worse than one that recommends fewer but better matches.

| Type | Metric | Why Netflix uses it |
|---|---|---|
| Offline | NDCG@K | Ranking quality on held-out watch history |
| Offline | AUC-ROC | Binary play-prediction quality |
| Online | **Streaming hours (north star)** | Did users actually watch more? Direct business metric |
| Online | Post-play survey | Did users report enjoying it? |
| Online | 30-day retention | Are members still subscribing? Long-term satisfaction |

**Why not CTR?** CTR can be gamed: mystery thumbnails maximise clicks, not satisfaction. Netflix's north star evolved from **CTR → streaming hours → member-satisfaction surveys**, as they learned that initial clicks do not predict long-term retention.

#### Spotify vs. Netflix freshness contrast
| | Candidate model | Ranker |
|---|---|---|
| Spotify | Weekly (batch) | N/A (one-stage) |
| Netflix | Daily (+ ANN rebuild) | Hourly |

#### Key lessons: student design vs. production reality
| Common student design | What Spotify / Netflix built | The lesson |
|---|---|---|
| Transformers or deep learning | word2vec (Spotify), two-tower MLP (Netflix) | Production favours **proven, scalable** architectures over the latest research |
| One model for everything | Candidate generation + separate ranker | **Separation of concerns** is not optional at 200M users |
| Train once, serve forever | Spotify weekly; Netflix hourly ranker | **Freshness strategy is a first-class architectural decision** |
| Cold-start as afterthought | Dedicated audio-CNN pipeline at Spotify | Cold-start affects every new song/user. Design for it explicitly |
| Optimise CTR | Netflix: streaming hours + post-play surveys | **Evaluation design determines what your model actually learns** |

**Scale forces separation of concerns:** at 200M users and 100M+ items, two-stage "is not an optimisation. It is the **only architecturally feasible option**." "The pattern you designed in W5S1 was correct."

### Demo notebook
None.

### Worksheet (w5s2_student_worksheet) — session structure and questions
| Phase | Time | Activity |
|---|---|---|
| 1. Compare | 0:00–0:10 | Answer 5 questions about your W5S1 design, **before any reveal** (honestly, not graded) |
| 2. Spotify | 0:10–0:50 | word2vec CF, audio-CNN cold-start, weekly batch pipeline |
| 3. Netflix | 0:50–1:20 | Two-tower candidate gen, separate ranker, streaming-hours evaluation |
| 4. Discussion | 1:20–1:45 | Fill in the compare table, share what surprised you |
| 5. Lessons | 1:45–2:00 | Five takeaways; preview of Week 6 (LLM serving) |

**Phase 1 questions:**
1. What algorithm did you use for candidate generation?
2. How often did your system retrain?
3. How did you handle a brand-new song?
4. What was your primary evaluation metric?
5. Did you separate candidate generation from ranking?

**Phase 2–3 fill-ins:**
- The word2vec analogy (playlists = sentences, songs = words; co-occurring songs end up *close*).
- What the audio CNN solves (new songs with no playlist data).
- What NLP-on-text captures (cultural discourse).
- Why Discover Weekly is weekly, and the product assumption behind that.
- The cold-start table.
- Draw the two-tower diagram.
- Why items can be pre-computed but users can't.
- Why a second model.
- The 3 ranker objectives.
- Why hourly for the ranker.
- What happens if you optimise CTR.
- Is your W5S1 metric offline or online/business?

**Phase 4 reflection:** what surprised you; what you got right; what you over-engineered.

**Phase 5, five lessons (verbatim-ish):**
1. **Production uses proven architectures, not the latest research.** word2vec (2013) and two-tower (2016) still run at scale in 2024.
2. **Freshness strategy is a first-class architectural decision.** Weekly vs. hourly determines infrastructure, cost and product experience.
3. **Evaluation design determines what your model actually optimises.** CTR as a north star produces clickbait.
4. **Cold-start is a product problem, not an ML afterthought.** Spotify built a dedicated pipeline for it.
5. **Scale forces separation of concerns.** At 200M users, one model cannot do everything.

**Week 6 preview question:** "LLM inference is fundamentally different from recommendation serving. What changes for: what is stateful vs. stateless, what is pre-computable vs. not, and the latency target for a conversational system?" Next week follows the same structure: S1 you design, S2 reveals what OpenAI and Cohere built.

### Pre/Post handout extras (w5s2_post_handout)

**Key takeaways (4):**
1. Playlists as sentences, songs as words: cultural co-occurrence without labels/audio/curation. "Songs that belong together cluster together because users put them together."
2. Cold-start is a first-class problem, not an edge case. The audio CNN gives a day-1 embedding. "Cold-start must be designed for explicitly. It cannot be added later."
3. Netflix optimises for long-term satisfaction, not CTR. "Choose badly and you build a system that harms the product it is supposed to improve."
4. Freshness strategy is an architectural decision. "Neither is right." It depends on business requirements, cost, and how fast user intent changes. "It cannot be bolted on after building the pipeline."

**Test Your Understanding, with official answers:**
- **Q1. Which embeddings does a new Spotify song get, in week 1 vs. after 30 days?** A: In the first week, **only an audio-CNN embedding** (from its spectrogram), because it has zero playlist co-occurrences and can't be in word2vec training. After ~30 days, it has accumulated enough playlist appearances to get a collaborative word2vec embedding, and serving **blends** both. The transition is **gradual, not sudden**.
- **Q2. Why is the Netflix ranker retrained hourly but the candidate model daily? What breaks if both are hourly?** A: Candidate gen builds an ANN index over all items. Rebuilding it is expensive (re-embed the whole catalogue). Hourly rebuilds would need massive GPU capacity and would often produce an unstable index. The ranker runs on only 1,000 candidates and has no index, so hourly retraining is cheap and captures within-day shifts. If the candidate model were hourly, the ANN index would be constantly stale during rebuilds, causing recommendation gaps, and the cost would be prohibitive.
- **Q3. PM wants "trending now" in Discover Weekly within 24h. What changes?** A: Add a **trending-signal fast path**: a real-time popularity counter (**Kafka + streaming aggregation**) that detects songs with rapidly accelerating play counts. Inject them **directly into the candidate set**, bypassing the weekly word2vec pipeline. **Trade-off:** you lose collaborative personalisation for those songs (everyone gets the same trending items), and you add popularity bias that can crowd out personalised recs if injection is too aggressive.
- **Q4. Two-tower vs. a matrix-factorisation baseline: one advantage and one disadvantage?** A: **Advantage:** separate user/item encoders let you **pre-compute item embeddings offline and add new items without retraining the full model**, and the approach scales to huge catalogues. **Disadvantage:** it can't capture arbitrary user-item interactions in the embedding stage, because the dot-product interaction is simple. MF is even simpler, can be faster to train and easier to debug on small data, but lacks the representation capacity of deep towers.

**Design exercises (no official answers):**
1. **Draw the Spotify pipeline** from memory. Annotate each box with (a) the data it reads, (b) the data it writes, (c) how often it runs. Compare with the case study.
2. **Freshness sensitivity analysis:** for a **news** article recommender, should you use weekly batch or hourly retraining? Cover (a) how fast intent changes in news vs. entertainment, and (b) cost/complexity, being concrete about what hourly retraining needs in infrastructure. (note: news goes stale within hours, so it leans hourly/streaming. Expect Kafka streaming features, frequent retrains, fast index updates, and more orchestration and monitoring cost.)
3. **Evaluation redesign for a short-video platform (TikTok-like):** propose three metrics: one easy to game (like CTR), one medium-term satisfaction (like streaming hours), one long-term health (like 30-day retention). For each, explain how a model could maximise it while harming users.

**Challenge, the Spotify podcast problem:** Spotify (after acquiring Gimlet Media) reused the music infra for podcasts. Give 3 reasons Discover Weekly doesn't transfer, each with a concrete engineering change. Consider: episode length and completion signals; content freshness (news podcasts vs. evergreen); one song played many times vs. one episode played once. (note: e.g. use completion % instead of play count as the signal; give news podcasts a recency/freshness fast path; don't re-recommend consumed episodes, so recommend *shows* plus the next episode instead.)

**Further reading:**
- Spotify Engineering Blog, "How Discover Weekly Works" (2015–2017; posts by Chris Johnson and the follow-up by Sophia Ciocca)
- Netflix Tech Blog on two-tower and multi-objective ranking
- **Covington, Adams, Sargin (2016), "Deep Neural Networks for YouTube Recommendations"**, RecSys 2016: the canonical two-stage candidate-gen + ranking paper
- **Mikolov et al. (2013)**, word2vec (NIPS 2013), Section 4

**Online resources:** Netflix Tech Blog, Spotify Engineering, Eugene Yan's blog (eugeneyan.com), RecSys papers (recsys.acm.org), **FAISS** (github.com/facebookresearch/faiss), **Annoy** (Spotify's ANN library, github.com/spotify/annoy).

### Likely doubts & exam angles
- **Q: How does Spotify get song embeddings without labels?** A: skip-gram word2vec over playlists, treated as sentences of song IDs. Co-occurrence gives 128-dim embeddings that capture cultural context.
- **Q: How does Spotify solve cold-start for new songs?** A: an audio CNN on the log-mel spectrogram gives a day-1 embedding. There is a separate audio-only index, and serving blends it with the collaborative index. Tiers: 0–7 d audio-only; 7–30 d audio + partial CF; 30+ d full ensemble.
- **Q: How is a user represented in Discover Weekly?** A: as the average of the embeddings of songs in their history. Candidates are scored by a weighted cosine similarity across 3 spaces, and the top 30 are returned.
- **Q: In Netflix's two-tower, what are the positives and negatives?** A: positive = title watched > 50%. Negatives = randomly sampled titles. Softmax or in-batch-negatives loss.
- **Q: Why separate the ranker from the two-tower?** A: the two-tower must be index-friendly (a simple dot product). The ranker sees only 1,000 items, so it can use expensive u×i interaction features and multiple objectives.
- **Q: What does the Netflix ranker optimise?** A: P(play), expected duration (> 75% watched), and post-play survey score.
- **Q: Why streaming hours over CTR?** A: CTR is gameable (clickbait, mystery thumbnails) and doesn't predict retention. The north star evolved CTR → streaming hours → satisfaction surveys, with 30-day retention also tracked.
- **Q: What is the different-cadence rationale?** A: candidate model daily (an ANN rebuild is expensive); ranker hourly (cheap, captures time-of-day/day-of-week intent).
- **Q: What does "simpler, not more complex" mean?** A: production uses word2vec (2013) and two-tower (2016). The value is in pipeline engineering: freshness, cold-start, evaluation.
- **Q: What is the trending fast-path trade-off?** A: freshness for viral songs, at the cost of personalisation for those items and added popularity bias.

---

## Cross-links
- **Rubric reuse:** the 10-step rubric (W5S1 Pt.1) and the "Framework Reference" page recur verbatim as page 1 of the W6S1 solution design, and in Weeks 7–8 case studies (W7, W8 solution designs). Same 10 questions, any problem.
- **Week 1 (ML systems 101 / design framework):** the rubric formalises the design framework from W1S2.
- **Week 2 (feature pipelines, model serving):** online/offline feature store (Redis + S3/Parquet), point-in-time correctness, training-serving skew, Triton serving and latency budgets come from W2S1/W2S2.
- **Week 3 (MLOps):** the MLflow registry stages, evaluation gates, canary/blue-green, Airflow DAGs, drift monitoring (Evidently) and the 4-layer monitoring stack are applied here.
- **Week 4 (RAG):** two-tower + ANN (FAISS) retrieval here is the same dense-retrieval idea used in RAG. W6S1 extends it to hybrid search + LLM augmentation.
- **Week 6:** same format. W5's "two-stage because of scale" reappears as "hybrid retrieval → reranker → LLM" in W6S1. The W5S2 preview contrasts batch recsys with synchronous, autoregressive LLM serving (W6S2).

## Glossary
- **10-step rubric:** Clarify, Context, Problem Statement, Requirements, Estimations, Architecture (a–f), Evaluation, A/B Testing, Deployment & Monitoring, Risk & Trade-offs.
- **Functional vs. non-functional requirements:** what the system does vs. how it must behave (latency, throughput, availability, scalability, privacy, maintainability).
- **p99 latency:** the 99th-percentile response time; the SLA unit (not the mean).
- **Back-of-envelope estimation:** rough sizing (QPS, storage, memory, pods). 2× error is fine; 1000× is a design error.
- **Two-stage architecture:** offline/cheap candidate generation (100M → ~1,000), then an expensive online ranker (1,000 → top-K).
- **Candidate generation:** retrieval stage that narrows the catalogue to a small candidate set.
- **Two-tower model:** separate user and item encoders trained jointly into a shared embedding space. Score = dot product, enabling ANN retrieval.
- **ANN (Approximate Nearest Neighbour):** fast approximate similarity search over embeddings (FAISS, Annoy, Pinecone).
- **FAISS:** Facebook AI Similarity Search library for ANN indices.
- **LightGBM:** gradient-boosted-tree library, used as the online ranker over tabular features.
- **Latency budget:** explicit per-step ms allowance summing to the SLA.
- **Implicit vs. explicit feedback:** plays/watches (abundant, noisy) vs. likes/ratings (sparse, < 2% of plays).
- **Point-in-time correct join:** training features use only data available before the label date. Prevents leakage.
- **Online / offline feature store:** Redis (low-latency serving) / S3 + Parquet (training history).
- **Precision@K / Recall@K / NDCG@K / AUC-ROC:** offline ranking and classification metrics.
- **North star metric:** the primary online business metric (CTR here; streaming hours at Netflix).
- **Guardrail metric:** a metric that must not degrade (diversity, latency, error rate, cold-start CTR gap).
- **Diversity score:** mean intra-list cosine distance of recommended items.
- **Novelty effect:** users engage more with anything new, which inflates short-term treatment metrics.
- **Selection bias / cannibalism / Simpson's paradox:** unbalanced groups / treatment affecting control / an aggregate trend reversing the subgroup trends.
- **Holdout group:** ~1% of users who never see new models, used for long-term measurement.
- **Power analysis:** sample-size calculation; smaller effects need bigger samples (5M users → 1% lift at α = 0.05, power 0.8).
- **Canary / blue-green / shadow traffic:** gradual % rollout / two parallel environments with a swap / mirrored traffic to the new version without serving its output.
- **Popularity bias / filter bubble / feedback loop:** rich-get-richer / users see only what they like / outputs become training data.
- **Cold-start (tiered fallback):** popularity + editorial → content-based → hybrid → full personalisation as interactions accumulate (0 / 1–4 / 5–19 / 20+).
- **Training-serving skew:** a feature-distribution mismatch between training and serving. Symptom: CTR drops while latency is unchanged.
- **word2vec (skip-gram):** learns embeddings from context co-occurrence. Spotify applies it to playlists (sentences) of songs (words).
- **Audio CNN:** CNN over log-mel spectrograms producing content embeddings; Spotify's new-song cold-start.
- **Log-mel spectrogram:** a time-frequency audio representation on the mel scale; the input to the audio CNN.
- **In-batch negatives:** using other items in the training batch as negatives for a contrastive/softmax loss.
- **Streaming hours:** Netflix north star, total watch time. Supplemented by post-play surveys and 30-day retention.
- **Trending fast path:** a real-time popularity counter (Kafka + streaming aggregation) that injects viral items into candidates.
