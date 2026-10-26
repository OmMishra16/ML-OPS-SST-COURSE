# Week 1 — Foundations: From Notebook to Production ML Systems, and the ML System Design Framework

Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week1/`):
- `w1s1_slides_ml_systems_101.pdf` (28 slides), `w1s1_pre_ml_systems_101.pdf`, `w1s1_post_ml_systems_101.pdf`, `w1s1_demo_ml_systems_101.ipynb`
- `w1s2_slides.pdf` (23 slides), `w1s2_pre_handout.pdf`, `w1s2_post_handout.pdf`, `w1s2_demo_ml_design_framework.ipynb`

Instructor: Priyansh Saxena, Scaler School of Technology, course "ML System Design and LLMOps". Course materials: https://www.priyanshsaxena.com/sst-ml-system-design
Primary text referenced throughout: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022). Session 2 is grounded almost entirely in Ch. 2.

(note: Slide 3 of Session 1 says "This course covers every box in depth over 12 weeks", while the pre-handout says "what the next 8 weeks look like". The course as delivered is 8 weeks; the "12 weeks" line looks like a leftover on the slide.)

---

## Session 1.1 — ML Systems 101: From Jupyter to Production
Subtitle: Production Gap • System Anatomy • Technical Debt • Design Framework. About 2 hours. Prerequisites: NLP & Transformers, Applied AI and ML.

Agenda (4 sections):
1. The Notebook-to-Production Gap
2. Anatomy of a Production ML System
3. Technical Debt in ML Systems
4. The ML System Design Framework

### TL;DR
- **Production ML is a system, not a model.** The model is typically about **5–10% of production code** (post-handout). The rest is data pipelines, feature engineering, serving infrastructure, monitoring and glue. "Everything between a working notebook and a reliable production system is **ML Systems Engineering**."
- **Six components** of a production ML system: Data Collection → Feature Store → Training Pipeline → Model Registry → Serving Layer → Monitoring, plus a **feedback / retrain trigger** loop from Monitoring back to training.
- **Three failure modes:** (1) training-serving skew, (2) silent degradation and data drift, (3) feedback loops and technical debt. ML fails *silently*: the model keeps returning confident, wrong predictions.
- **Training-serving skew is "the #1 silent killer".** The fix is to save the whole preprocessing pipeline (for example `sklearn.pipeline.Pipeline`), not just the model weights.
- **Sculley et al. 2015 (Hidden Technical Debt in ML Systems, NeurIPS):** ML code is a small fraction of the system. ML-specific debt includes pipeline jungles, glue code, correction cascades, undeclared consumers and dead experimental codepaths. "ML debt compounds faster than software debt because data dependencies are invisible."
- **The 5-Question Framework:** What problem? → What data? → What model? → How to serve? → How to monitor? Answer all five before writing code; "skipping any one of them guarantees a rebuild later."

### Concepts (in lecture order)

#### §1 The Notebook-to-Production Gap

##### "You trained a model. Now what?" (slide 2)
| What you built | What production needs |
|---|---|
| A Jupyter notebook | 24/7 availability (99.9% uptime) |
| Accuracy: 94% on the test set | Latency SLA ≤ 100 ms p99 |
| Runs on your local GPU | Input validation & error handling |
| `model.pkl` sitting in `/tmp` | Monitoring and alerting |
| You: "It works!" | Automated retrain pipeline |
| | Rollback strategy |

**The gap:** everything between a working notebook and a reliable production system is ML Systems Engineering.

##### The six components of a production ML system (slide 3)
```
Data Collection → Feature Store → Training Pipeline → Model Registry → Serving Layer → Monitoring
                                        ^                                              |
                                        |------------ feedback / retrain trigger ------|
```
- Each component is **independently deployable, versioned, and monitored**.
- A failure in **any** component can **silently** degrade model performance.

##### Failure Mode 1: Training-Serving Skew (slide 4)
- **What happens:** you fit a `StandardScaler` on training data, forget to save it alongside the model, and raw (unscaled) features go in at serving time. Predictions are garbage, **silently**.
- **Root cause:** feature preprocessing is not part of the saved artefact, and training and serving code diverge over time.
- **The fix:**
  - Save the **entire preprocessing pipeline**, not just model weights.
  - Use `sklearn.pipeline.Pipeline` or an equivalent.
  - Validate feature statistics at serving time.
  - Assert a schema match: training ≡ serving.

##### Failure Mode 2: Silent Degradation & Data Drift (slide 5)
- **What happens:** a model trained in Jan 2020 is deployed in Feb 2020 with no monitoring and no retraining schedule. The world changes (user behaviour, market, language). Accuracy quietly falls from **94% to 61%**, and **nobody notices for 6 months**.
- **Types of drift:**
  | Drift type | What shifts |
  |---|---|
  | **Data drift** | input distribution **P(X)** |
  | **Concept drift** | **P(Y\|X)**, the relationship between inputs and the label |
  | **Label shift** | output distribution **P(Y)** |
- **Real example:** travel demand models trained pre-COVID and deployed in 2021. P(travel | flight search) collapsed overnight. With no monitoring, recommendations were bad for months.
- **The fix:** monitor input feature distributions **daily**, track the prediction distribution against a baseline, and set up **automated retrain triggers**.

##### Failure Mode 3: Feedback Loops & Technical Debt (slide 6)
- **Feedback loops:** the model's predictions become future training data. Example: a recommender shows only pop music, so only pop music gets clicked, so the model learns to show more pop. The result is self-reinforcing bias with no external correction signal. This is called **exposure bias** or **closed-loop amplification**.
- **Technical debt (Sculley et al. 2015):**
  - **Glue code:** 500 lines of hacky wrappers connecting components.
  - **Pipeline jungles:** no one knows what feeds what.
  - **Dead experimental codepaths:** "we might need this later".
  - **Undeclared consumers:** models silently reading each other's outputs.
- **Key insight:** "ML debt compounds faster than software debt because data dependencies are invisible."

##### Quiz 1 (slides 7–8)
- **Q1 (Amazon hiring algorithm):** a resume-screening model trained on historical hiring data downranked resumes containing "women's" (for example, "women's chess club"). Which failure mode is this?
  **A:** A **feedback loop / training data bias**. The model was trained on historical human hiring decisions that systematically favoured male candidates, and it learned and amplified that bias. This is classic **closed-loop amplification**: biased labels become biased predictions.
- **Q2 (travel prediction):** a model trained on 2018–2019 data and deployed in March 2021 was wildly off for months before anyone noticed.
  **A:** **Concept drift (silent degradation).** COVID-19 caused P(Y|X), the relationship between search signals and actual travel, to collapse overnight. It should have been caught by **monitoring the prediction distribution (output drift)** and **business metrics** (actual booking rates). A **daily drift alert on feature distributions** would have fired within days of lockdowns starting.

#### §2 Anatomy of a Production ML System
"Real systems rarely look this clean — but understanding each piece lets you reason about the mess." For each component the slides give its purpose, key concerns and tooling.

##### Data Collection (slide 10)
- **Ingestion patterns:**
  - **Event streams** (Kafka, Kinesis) for real-time user actions.
  - **Batch ingestion**: nightly ETL from the data warehouse.
  - **Third-party APIs**: weather, financial feeds.
  - **Human labels**: crowd-sourced annotation pipelines.
- **Key decisions:** How often to collect (frequency vs. cost)? Where to store raw vs. processed data? Who owns schema evolution?
- **Data quality checks (non-negotiable):**
  - **Schema validation**: expected columns, dtypes.
  - **Null/missing rate**: per-column thresholds.
  - **Range checks**: e.g. age ∈ [0, 130].
  - **Referential integrity**: foreign keys exist.
  - **Volume checks**: a sudden drop means a pipeline failure.
- **Tools:** Great Expectations, dbt tests, Soda.

##### Feature Store (slide 11)
| Offline store (training) | Online store (serving) |
|---|---|
| Historical feature values for training | Low-latency lookup at inference time |
| Column-oriented format (Parquet, Delta Lake) | Key-value store (Redis, DynamoDB) |
| **Point-in-time correctness**: the value at t uses only data available before t | Must stay **in sync** with the offline store |
| Prevents **data leakage** | **Staleness budget**: how old is too old? |

- **Why not just recompute features at serving time?** Because of **cost, latency, and reproducibility**. A feature store lets you "compute once, use everywhere", so training and serving use **identical feature logic**.
- **Tools:** Feast (open source), Tecton (managed), Hopsworks.

##### Training Pipeline (slide 12)
- **What it does:** pulls versioned data from the feature store, runs preprocessing and feature engineering, trains with tracked hyperparameters, evaluates on a held-out validation set, **pushes the artefact to the model registry if metrics pass**, and **triggers retraining on a schedule or on a drift alert**.
- **What must be tracked:**
  - **Data version**: which dataset snapshot.
  - **Code version**: git commit hash.
  - **Hyperparameters**: every knob.
  - **Metrics**: train, val and test, not just accuracy.
  - **Artefacts**: model file, preprocessor, config.
- **Tools:** MLflow, Weights & Biases, DVC, Kubeflow Pipelines.

##### Model Registry (slide 13): "More than a file on S3"
- A centralised catalogue of trained models. Each model has a **version, lineage, metrics and tags**.
- **Lifecycle states:** Staging → Production → Archived.
- It promotes collaboration: the data scientist trains, the platform team deploys.
- It enables **instant rollback**: demote the bad model and promote the previous one.

| S3 bucket | Model registry |
|---|---|
| File storage | Metadata + lineage |
| No versioning | Semantic versions |
| No lifecycle | Staging/Prod/Archive |
| Manual discovery | Search & filter |
| No audit trail | Full audit log |

- **Tools:** MLflow Registry, W&B Registry, Vertex AI Model Registry.

##### Serving Layer (slide 14)
- **Serving patterns:**
  - **Synchronous REST/gRPC**: the client waits. Use when latency < 1 s is required.
  - **Async batch**: processes queued jobs. Use for offline scoring.
  - **Streaming**: continuous inference on an event stream. Use for real-time pipelines.
  - **Edge**: the model runs on the device. Use for privacy or offline operation.
- **Latency budgets:**
  - **Define the SLA before choosing the model size.**
  - Measure **p50, p95, p99, not the mean**.
  - **Budget = feature lookup + preprocessing + inference + postprocessing.**
  - Large model + no caching + slow feature lookup = SLA breach.
- **Tools:** BentoML, Triton Inference Server, TorchServe, Ray Serve.

##### Monitoring (slide 15)
- **What to monitor:**
  - **Input drift**: are features shifting from the training distribution?
  - **Output drift**: is the prediction distribution changing?
  - **Data quality**: nulls, schema violations, volume drops.
  - **Business metrics**: click rate, conversion, revenue impact.
  - **System metrics**: latency, error rate, throughput.
- **Alerting hierarchy** (in order):
  1. **Data quality alert**: the pipeline may be broken.
  2. **Input drift alert**: consider retraining.
  3. **Output drift alert**: investigate urgently.
  4. **Business metric drop**: a model or product issue.
- **Tools:** Evidently AI, Arize, WhyLabs, Prometheus + Grafana.

##### Quiz 2 (slides 16–17)
- **Q1 (offline vs. online feature store):** a fraud model uses "number of transactions by this user in the past 7 days". At serving time, should this come from the offline or the online store? Bonus: what goes wrong if you compute it fresh from the DB on every request?
  **A:** The **online store**. At serving time you need the feature instantly, so it must be pre-computed and cached in a key-value store such as Redis. **Bonus:** computing it fresh from the DB on every request adds **50–200 ms** of query latency, risks an SLA breach, and creates **training-serving skew** if the DB query logic differs from how the feature was computed at training time.
- **Q2 (registry vs. S3):** the team stores models as `model_v3_final_FINAL.pkl`. Name two problems this causes in a team of 5 data scientists.
  **A:** (1) **No reproducibility**: there is no lineage, so you cannot tell which dataset or code version produced the model, and debugging a production regression becomes impossible. (2) **No rollback**: there are no lifecycle states, so reverting means manually hunting for the previous file. **The registry fix:** semantic versions + lineage metadata + one-click promotion/demotion.

#### §3 Technical Debt in ML Systems

##### Hidden Technical Debt in ML Systems: Sculley et al., Google (NeurIPS 2015) (slide 18)
- A seminal paper from Google's production ML team.
- **Key finding: ML code is a small fraction of a production ML system.** Most of the code is infrastructure, pipelines and glue.
- The diagram (adapted from Sculley) shows a small "ML Code" box surrounded by Data Collection, Feature Engineering, Serving Infra, Monitoring, Config Mgmt and Process Mgmt.
- All of it accrues debt, but ML debt is harder to see.
- **Data dependencies are "entanglement"**: changing one input changes everything. (note: this is Sculley's CACE principle, "Changing Anything Changes Everything".)

##### Pipeline Jungles (slide 19)
- **How it starts:** Team A writes a pipeline for feature f1. Team B copies and modifies it. Now 3 versions exist and nobody knows which one feeds prod. "Undeclared consumers" read the outputs silently.
- **Signs of a pipeline jungle:**
  - No one can draw the data flow end-to-end.
  - Deleting an old table breaks something unexpected.
  - Reprocessing old data gives different results.
- **Real cost:** at one large company, **30% of ML engineer time** went into understanding existing pipelines rather than building new ones (Sculley et al.).
- **The fix:** explicit **data lineage tracking**, a **single source of truth per feature** (a feature store), **ownership tags** on every pipeline, and **regular audits** that delete dead pipelines.

##### Glue Code & Correction Cascades (slide 20)
- **Glue code:** a generic open-source model wrapped in 1000 lines of custom glue. The glue handles edge cases, format conversion, schema mismatches and post-processing hacks. It makes swapping the model nearly impossible, because **the glue, not the model, carries the real logic**.
- **Correction cascades:** Model A is slightly wrong for edge case e, so the team adds Model B on top to fix e. Model B then errs on edge case f, so the team adds Model C, and so on. The cascade A → B → C → … means that changing A breaks B, C and everything after.
- **The danger of both:** neither shows up in model accuracy metrics. They only surface when you try to change something. **The cost is velocity loss: every new feature takes 3× longer.**

##### The Debt Meter (slide 21)
| Debt pattern | Where it comes from | How to fix it |
|---|---|---|
| Training-serving skew | Preprocessing not saved with model; serving code diverges from training code | Use Pipeline objects; validate feature stats at serving time |
| Pipeline jungle | Organic growth, no ownership, undeclared consumers | Data lineage tool; feature store; ownership tagging |
| Glue code | Wrapping general-purpose libraries without abstraction | Refactor to clean API; replace glue with proper integration layer |
| Correction cascades | Patching model outputs instead of fixing root cause | Fix upstream model or data; avoid chaining model outputs |

Break prompt: "Think about a model you have built or used. **Which failure mode would bite it first?**"

#### §4 The ML System Design Framework

##### The 5-Question Framework (slide 23)
| # | Question | What to answer |
|---|---|---|
| Q1 | **What problem?** | ML objective, success metric, business value |
| Q2 | **What data?** | Sources, volume, freshness, labelling cost, legal constraints |
| Q3 | **What model?** | Complexity vs. latency vs. interpretability trade-offs |
| Q4 | **How to serve?** | Sync vs. async, latency SLA, scaling requirements |
| Q5 | **How to monitor?** | What breaks first? How do you detect it? **Who gets paged?** |

This framework is applied to every system designed in the course. Session 1.2 goes deep on Q1.

##### Applying the framework: Spam Classifier (slide 24)
| Question | Answer for spam detection |
|---|---|
| 1. What problem? | Binary classification: is this email spam? Success metric: **precision ≥ 99%** (false positives delete real email), **recall ≥ 95%** |
| 2. What data? | Email text + metadata (sender, subject, headers). Labels: user "mark as spam" actions. **500M emails/day**, retention **90 days**. **GDPR applies.** |
| 3. What model? | Start with **logistic regression on TF-IDF** (fast, interpretable, low latency). Iterate to **fine-tuned BERT** if precision is insufficient. |
| 4. How to serve? | **Synchronous REST, ≤ 50 ms SLA** (an email arrives and is classified before inbox delivery). Deploy as a **sidecar to the mail server**. |
| 5. How to monitor? | Track the % of emails users mark as spam after delivery (**false negatives**) and the % of spam complaints on ham (**false positives**). **Retrain weekly.** |

##### Trade-off 1: Latency vs. Accuracy (slide 25)
- This is "the fundamental serving tension". Larger models give higher accuracy and higher latency.
- **Quantisation, distillation and pruning** recover accuracy at lower cost.
- **The latency SLA drives the model choice, not the other way round.**
- **Caching:** a cache hit means 0 ms latency.
- The chart plots accuracy against latency with BERT (high/high), DistilBERT (middle) and LR+TF-IDF (low/low) along a **Pareto frontier**. Points off the frontier are "suboptimal".
- **Rules of thumb:**
  | Setting | Budget | Choice |
  |---|---|---|
  | Mobile / real-time | **< 50 ms** | small model, edge |
  | API product | **< 300 ms** | medium model, GPU |
  | Batch overnight | no SLA | biggest affordable model |

##### Trade-off 2: Freshness vs. Cost (slide 26)
- **How often to retrain?** Retraining is not free: it costs compute and engineering time, and carries **regression risk**.
- **The staleness curve is domain-specific:**
  - News recommendation degrades in **hours**.
  - Credit scoring degrades over **months**.
  - Medical imaging is stable for **years**.
  - (The chart plots model quality against time since training, with an SLA threshold line. The news curve crosses it fast; the credit curve crosses it slowly.)
- **Retraining triggers:**
  1. **Scheduled**: weekly or daily, regardless of drift.
  2. **Drift-triggered**: a statistical test fires when P(X) shifts.
  3. **Performance-triggered**: a metric drops below a threshold.

##### When to Use Which Serving Pattern (slide 27)
| Pattern | When to use | Examples |
|---|---|---|
| **Synchronous REST** | User is waiting; latency < 1 s; single item at a time; simple scaling | Spam filter, fraud check at checkout, search ranking |
| **Asynchronous batch** | High volume, no real-time need; overnight or hourly runs; cost-sensitive | Monthly churn prediction, weekly recommendation refresh, nightly demand forecast |
| **Streaming (Kafka + Flink)** | Continuous event stream; sub-second freshness; stateful aggregations across events | Real-time fraud detection, dynamic pricing, live anomaly detection |
| **Edge / on-device** | Privacy constraints; offline use; ultra-low latency (**< 10 ms**); no cloud cost | Face unlock, keyword spotting, autonomous vehicle perception |

Closing line: "Shipped to prod at 2am. Monitoring optional. What could go wrong?"

### Demo notebook (`w1s1_demo_ml_systems_101.ipynb`)
The instructor runs the live demo in three parts, plus a discussion.

**Part 1: The notebook version** ("We trained a model. It got 94% accuracy. Let's ship it.")
- `load_breast_cancer`, `train_test_split(test_size=0.2, random_state=42)`, `StandardScaler`, and `LogisticRegression(max_iter=1000)`. The notebook prints test accuracy, then "Ship it?".
- Checklist of what was and wasn't done: ✅ trained a model, ✅ evaluated on a test set; ❌ saved the scaler, ❌ saved the model, ❌ versioned either of them, ❌ wrote a serving function, ❌ **defined what "good" looks like at month 3**.
- Key line: **"The question isn't 'does it work?' — it's 'does it keep working?'"**

**Part 2: Training-serving skew demo**
```python
pickle.dump(model, f)                  # saves ONLY the model, not the scaler
loaded_model.predict(X_test)           # raw, unscaled input: no error, wrong predictions
```
- The demo prints the original (scaled) accuracy, the serving (unscaled) accuracy and the "silent accuracy drop", followed by "No error was raised. No exception. No alert."
- **Why it happens:** the model was trained on standardized features (mean = 0, std = 1) but received raw features at serving time (mean ≈ 14, std ≈ 7 for some columns). The model had no way to know.
- "This is training-serving skew. **It is silent. It is common. It is preventable.**"
- **The fix:**
```python
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
pipeline.fit(X_train, y_train)
pickle.dump(pipeline, f)   # raw input in -> correct prediction out; skew eliminated
```

**Part 3: A six-component ML system sketched in Python** (toy, with in-memory stand-ins)
| Component | Demo stand-in | "In prod" |
|---|---|---|
| 1. Data pipeline | `data_pipeline(user_id)` returns a raw dict; `assert "user_id" in raw` acts as schema validation | reads from Kafka/S3 |
| 2. Feature store | a `feature_store = {}` dict with `update_feature_store` / `get_features` | Redis (online) + S3/Parquet (offline) |
| 3. Training pipeline | `train_model(version)` fits a Pipeline and puts it in `model_registry[version]` | scheduled Airflow/Prefect job |
| 4. Model registry | `promote_to_production(version)` sets a global `production_model_version` | MLflow Model Registry |
| 5. Serving layer | `serve(raw_features)` asserts that a prod model exists, predicts, returns `{prediction, confidence, model}`, and appends to `prediction_log = deque(maxlen=100)` | REST/gRPC behind a load balancer |
| 6. Monitoring | `monitor()` reports the prediction count, average confidence and active model, and warns "LOW CONFIDENCE — investigate feature drift" if avg confidence < 0.7 | runs on a schedule, writes to dashboards |

- Run order: ingest → train + register v1 → promote v1 → serve 5 predictions → monitor.
- (note: `serve()` is deliberately simplified. It ignores the passed features and predicts on `data.data[0]`, so every prediction is the same.)

**Part 4: Discussion questions**
- "What would break first if we deployed this to 1 million users?"
- "What's missing from our `monitor()` function?" Hint: we track confidence, so what else matters? (note: grounded in slide 15, the missing items are input drift, output/prediction-distribution drift, data quality, business metrics, system metrics such as latency, errors and throughput, and actual ground-truth performance.)

### Pre/Post handout extras
**Pre-handout (what to expect, how to prepare):**
- An opening discussion on how you would deploy your best model right now, the six-component anatomy diagram, two short quizzes on failure modes, a Sculley et al. discussion, a live tooling demo, and the course roadmap.
- Preparation: recall where your NLP features "lived" at serving time, optionally skim the Sculley 2015 abstract, and have Colab with a standard runtime. No new tools are needed: "we are setting the mental model first".

**Post-handout key takeaways:** production ML is a system (the model is 5–10% of the code); training-serving skew is the #1 silent killer, so log and monitor feature distributions at **serving** time, not just at training time; technical debt compounds fast and should be addressed at design time; answer the 5 questions before writing code.

**Test Your Understanding (with handout answers):**
- **Q1.** A model trained on last month's data is deployed. Three months later accuracy has dropped 15% with no errors logged. What failure mode is this, and how do you detect it before users notice?
  **A:** **Data drift** (the world changed, the model did not) **combined with silent degradation** (no error logs, because the predictions are valid, just wrong). Detect it by monitoring feature distributions at serving time (for example with Evidently) and tracking business metrics such as conversion and churn alongside model metrics.
- **Q2.** Five scripts read the same feature table, only 2 update it, and nobody documented which. What does Sculley call this?
  **A:** **Undeclared consumers**: multiple downstream consumers depend on a source with no single owner and no documented contract. A schema change silently breaks the other scripts.
- **Q3.** Apply the 5 questions to fraud detection.
  **A:** Problem: detect fraudulent transactions before they complete. Data: transaction events with user history, merchant data and device fingerprint. Model: a binary classifier with a **< 100 ms** latency requirement. Serving: real-time synchronous, embedded in the payment flow. Monitoring: track the **false positive rate alongside the fraud catch rate**, because both matter to the business.
- **Q4.** Model registry vs. S3: name two things the registry gives you.
  **A:** (1) Versioning with metadata (who trained it, on what data, with what metrics), whereas S3 gives you a filename. (2) Lifecycle management (staging/production/archived states with promotion workflows). Also an audit trail and one-command rollback.

**Exercises:**
1. **Audit a notebook:** take a past notebook, identify which of the six components are missing, and write a one-paragraph production gap analysis.
2. **5-Question design doc** for (a) Spotify song recommendation, (b) the Gmail spam filter or (c) Uber surge pricing: one page. "Focus on **data and serving** — those are where most designs go wrong."
3. **Find your own technical debt:** read Sculley §2 and name two debt patterns you have created, with one sentence each on how to fix them.
- **Challenge, the ML postmortem:** pick Amazon hiring (2018), Zillow iBuying (2021) or Uber surge pricing. Write (1) what went wrong *in ML-system terms*, (2) which component(s) failed, (3) what monitoring would have caught it, and (4) what you would design differently.

**Further reading:** Sculley et al. 2015 (read §§1–3); Chip Huyen DMLS Ch. 1–2 (the course primary text: Ch. 1 is the landscape, Ch. 2 is scoping); Karpathy "Software 2.0" (2017); Eugene Yan "Patterns for Building LLM-based Systems" (2023, relevant from Week 4). Online: MLflow docs (used from Week 3), Evidently AI blog, Google ML Crash Course systems chapter, Chip Huyen's blog, DAIR.AI ML News, The Batch.

### Likely doubts & exam angles
1. **Data drift vs. concept drift vs. label shift?** Data drift is P(X) shifting; concept drift is P(Y|X) shifting (the same inputs now mean a different outcome, as with COVID travel); label shift is P(Y) shifting.
2. **Why is training-serving skew "silent"?** The model still returns confident, valid-looking predictions, so there is no exception or error log. You only catch it by validating feature statistics and schema at serving time.
3. **Online vs. offline feature store?** Offline holds historical values in a columnar store (Parquet/Delta) for training, with point-in-time correctness to prevent leakage. Online is a low-latency KV store (Redis/DynamoDB) for inference, kept in sync with offline and subject to a staleness budget.
4. **Why measure p99 rather than mean latency?** SLAs are about tail latency. The latency budget is feature lookup + preprocessing + inference + postprocessing.
5. **Which serving pattern for X?** The user is waiting → sync REST; high volume with no real-time need → async batch; continuous events with stateful aggregation → streaming (Kafka + Flink); privacy, offline use or < 10 ms → edge.
6. **What is the order of the alerting hierarchy?** Data quality → input drift → output drift → business metric drop.
7. **Which three retraining triggers are there?** Scheduled, drift-triggered and performance-triggered. The right frequency depends on the domain's staleness curve: hours for news, months for credit, years for medical imaging.
8. **Name 4+ Sculley debt patterns.** Glue code, pipeline jungles, dead experimental codepaths, undeclared consumers, correction cascades, and data dependency entanglement.
9. **Amazon hiring is which failure?** A feedback loop / training data bias (closed-loop amplification).
10. **Spam SLA numbers?** Precision ≥ 99%, recall ≥ 95%, ≤ 50 ms sync REST as a mail-server sidecar, 500M emails/day, weekly retraining, GDPR. Precision is prioritised because false positives delete real email.

---

## Session 1.2 — The ML System Design Framework
Subtitle: Business Objectives • Four Requirements • Iterative Process • Problem Framing. About 2 hours. Source: Chip Huyen, *Designing ML Systems*, Ch. 2.

Agenda: (1) From ML Metrics to Business Metrics, (2) Four Requirements for ML Systems, (3) The Iterative Development Process, (4) Framing ML Problems, (5) Mind vs. Data.

The session goes deep on **Q1 ("What problem?")** of the 5-question framework.

### TL;DR
- **Business metrics first, always.** "Most companies don't care about ML metrics unless they move business metrics" (Chip Huyen). Every ML metric needs a traceable path to a business outcome. If you can't draw that path, reconsider whether ML is the right solution.
- **Four requirements drive architecture:** Reliability, Scalability, Maintainability, Adaptability. Ignoring any one guarantees a costly rebuild.
- **6-step iterative cycle** (never-ending): Project Scoping → Data Engineering → ML Model Development → Deployment → Monitoring & Continual Learning → Business Analysis → back to scoping. The arrows go backwards too.
- **An ML problem = Inputs + Outputs + Objective function.** "Slow customer support" is a business problem, not an ML problem, until you frame it.
- **Framing changes everything.** Next-app prediction framed as classification needs a retrain for every new app; framed as regression/scoring it needs none. **Decouple objectives:** use separate models and blend them at serving time, so PMs can tune α/β without retraining.
- **Mind vs. Data:** Rogati's AI hierarchy of needs, plus Norvig, Sutton (The Bitter Lesson) and Pearl. "Algorithmic improvements are what win headlines. Data improvements are what win production."

### Concepts (in lecture order)

#### §1 From ML Metrics to Business Metrics

##### Why ML Projects Get Killed (slide 2)
- **The pattern Chip Huyen observes:** data scientists optimise ML metrics (loss, AUC, F1), business stakeholders track revenue, retention and cost, and the two groups talk past each other. The project is cancelled: "we spent 6 months and nothing changed".
- **Key insight:** "Most companies don't care about ML metrics unless they move business metrics." (Chip Huyen, DMLS Ch. 2)
- **The right questions to ask first:**
  1. What business metric should improve?
  2. By how much, and in what timeframe?
  3. How does the ML metric connect to it?
  4. What is the cost of being wrong?
- **Rule of thumb:** every ML metric you optimise should have a clear, traceable path to a business outcome. If you cannot draw that path, reconsider whether ML is the right solution.

##### Translating Business Metrics to ML Metrics (slide 3)
| Business metric | What it measures | Proxy ML metric |
|---|---|---|
| Monthly Active Users | Engagement and product stickiness | Predicted session length, click-through rate |
| Subscription cancellation rate | Long-term retention | Predicted churn probability |
| Revenue per user | Monetisation effectiveness | Predicted purchase probability, basket size |
| Customer support cost | Operational efficiency | Auto-resolution rate, ticket routing accuracy |
| Fraud losses ($) | Risk exposure | Precision/recall of fraud classifier at fixed threshold |

**The proxy ML metric should move in the same direction as the business metric. Always verify this empirically.**

##### Netflix Take-Rate: A Worked Example (slide 4)
- Netflix runs a recommender serving billions of impressions/day. Which metric should the team optimise: raw accuracy on held-out ratings? NDCG?
- **Netflix's answer is take-rate:**
  **take-rate = # quality plays / # recommendations shown**
- **Causal chain:** Better recommender → Higher take-rate → More streaming hours → Lower cancellation rate.
- Source: Chip Huyen, DMLS Ch. 2.

##### "Magically: Possible. Overnight: No." The deployment maturity curve (slide 5)
- ROI on ML depends heavily on **organisational maturity**.
  - Among companies with models in production for **5+ years**, **75% can deploy a new model in under 30 days**.
  - Among companies **just getting started**, **60% take over 30 days to deploy their first model**.
- **What maturity unlocks:** reusable data pipelines (no re-ingestion from scratch), a shared feature store (no feature recomputation), an automated training pipeline (no manual runs), one-click deployment via the model registry, and monitoring that is already wired up.
- **Implication:** at a company with no ML infrastructure, your first project will take longer than you expect. Plan for it and don't promise "overnight magic".
- **Practical advice:** invest in infrastructure early. Each project after the first gets faster, but only if you build the foundations right.

#### §2 Four Requirements for ML Systems

##### Reliability: When ML Fails Silently (slide 6)
| Traditional software failures | ML system failures |
|---|---|
| Crash with a stack trace | Model returns a prediction, confidently |
| 500 error code to the user | No exception, no error log |
| Clear signal: something broke | Prediction is just wrong |
| On-call engineer paged immediately | Nobody notices for weeks |

- **The Google Translate problem:** "If you use Google Translate to translate a sentence into a language you don't know, it might be very hard for you to tell even if the translation is wrong." (Chip Huyen, Ch. 2)
- **What reliability requires:**
  - Monitoring **prediction distributions**, not just uptime.
  - **Proxy metrics** that correlate with output quality.
  - **Fallback strategies**: a rule-based backup, or the previous model version.

##### Scalability: 8,000 Models for 8,000 Customers (slide 7)
- **Scalability is not just traffic.** There are three axes of scale: **Complexity** (one model vs. thousands), **Traffic** (100 requests/day vs. 100 million), and **Model count**.
- **Model count:** a startup Chip Huyen worked with had **8,000 models, one per enterprise customer**. Each model needs its own training data, retraining schedule, monitoring and deployment.
- **Amazon Prime Day failure:** Amazon's autoscaling failed during Prime Day at a cost of **$72–99M in a single hour**. "Scalability is not a nice-to-have."

##### Maintainability and Adaptability (slide 8)
- **Maintainability:** multiple contributors (ML engineers, DevOps, subject-matter experts) must work on the same system, and each group has different tools, languages and priorities. Requirements:
  - Code is documented and reviewable.
  - **Data is versioned** (not just "latest.csv").
  - **Models are reproducible: same code + same data = same model.**
  - Without maintainability you cannot onboard new engineers or debug production issues.
- **Adaptability:** data distributions shift over time (users change behaviour) and business requirements change (new markets, regulations). The system must adapt **without service interruption**. This is different from reliability; you need both.
  - What adaptability looks like: automated retraining pipelines, **canary deployments** for new models, **feature flag infrastructure** for quick rollbacks, and regular model evaluation cadences.

##### The Four Requirements: Summary (slide 9)
| Requirement | What it means | How it fails without it |
|---|---|---|
| **Reliability** | Performs correctly even in the face of adversity: bad inputs, infrastructure failures, edge cases | Predictions degrade silently; users and engineers have no idea until business metrics drop |
| **Scalability** | Handles growth in traffic, model complexity and number of models without architectural rework | Breaks under load; Prime Day failures; 8K models become unmanageable |
| **Maintainability** | Multiple teams can understand, modify and extend the system independently | Only the original author can change anything; onboarding takes months; bugs are impossible to trace |
| **Adaptability** | Evolves with shifting data distributions and business requirements without service interruption | Model goes stale; adapting requires a full system rebuild; the business loses its competitive edge |

Break slide: "Your model is reliable, scalable, maintainable, and adaptable. Congratulations — it also hasn't been deployed yet."

#### §3 The Iterative Development Process

##### The 6-Step Development Cycle (slide 11)
```
1. Project Scoping → 2. Data Engineering → 3. ML Model Development → 4. Deployment
        ^                                                                   |
6. Business Analysis ←——— 5. Monitoring & Continual Learning ←—————————————┘
   (→ back to 1)
```
- "Developing an ML system is an iterative and, in most cases, **never-ending** process." (Chip Huyen, Ch. 2)
- **The arrows do not only go forward.** Business analysis often sends you back to scoping, and monitoring findings routinely trigger new data engineering work.

##### What Each Step Really Involves (slide 12)
1. **Project Scoping:** define the business objective and the ML objective, identify stakeholders and constraints, estimate data availability and labelling cost, and **define success criteria before any modelling**.
2. **Data Engineering:** data collection, cleaning and labelling pipelines; feature engineering and feature store population; data versioning and lineage tracking.
3. **ML Model Development:** a baseline model (**always start simple**), iterative experimentation with tracked metrics, and evaluation against held-out data **and business criteria**.
4. **Deployment:** package the model **with its preprocessing pipeline**, choose a serving pattern (sync, async, edge), and run in **shadow mode or a canary rollout** before full launch.
5. **Monitoring & Continual Learning:** track input drift, output drift and business metrics; trigger retraining when thresholds are breached; incorporate new labelled data continuously.
6. **Business Analysis:** Did the business metric move? Is the model worth the ongoing infrastructure cost? What should the next iteration focus on?

##### Quiz 1 (slides 13–14)
- **Q1 (the cycle is not linear):** after step 3 the team discovers the training data is missing a critical signal, location at the time of purchase. Where do they go back to?
  **A:** **Step 2, Data Engineering.** Collect the missing location signal, add it to the feature store, re-run the labelling pipeline, and version the new dataset. **It could have been caught at Step 1 (Project Scoping)**, where a proper data audit and feature requirement checklist would have identified location as critical before modelling.
- **Q2 (business analysis feedback):** a churn model reaches 87% AUC, but after deployment the cancellation rate doesn't change. Did the model fail?
  **A:** **Not necessarily. The *intervention* may have failed.** 87% AUC means the model ranks churners correctly, but a prediction alone does not reduce churn. Step 6 asks: (a) Was the output actually used to trigger a retention action (discount, outreach)? (b) Was the action effective? (c) Is AUC the right metric, or should we optimise for **actionable churn risk at a precision that makes retention cost-effective**?

#### §4 Framing ML Problems

##### What Makes Something an ML Problem? (slide 15)
- An ML problem has **three parts**:
  1. **Inputs**: what data does the model receive?
  2. **Outputs**: what should the model predict or generate?
  3. **Objective function**: how do we measure how good the output is?
- **A common mistake:** "Slow customer support" is not an ML problem. It is a business problem, and you must frame it.

| Part | Customer support routing |
|---|---|
| Business problem | Support is slow |
| ML task | Route requests to 4 depts |
| Input | Customer request text |
| Output | Department label |
| Objective | Minimise misrouting rate |
| Task type | Multiclass classification |

The business problem becomes solvable once you have inputs, outputs and an objective function.

##### Task Type Taxonomy (slide 16)
| Task type | Output | Loss | Examples / notes |
|---|---|---|---|
| **Regression** | Continuous value | RMSE, MAE | House price, demand forecast, ETA |
| **Binary classification** | One of two classes (yes/no, fraud/legit) | Logistic loss (binary cross-entropy) | Spam filter, churn prediction, fraud detection |
| **Multiclass classification** | One of N classes | Cross-entropy over N classes | Product category, language detection. **Rule of thumb: you need ~100,000 examples for 1,000 classes** |
| **Multilabel classification** | Each input can belong to multiple classes at once | (note: typically per-label binary cross-entropy) | Document tagging, multi-genre movie labels. **Hardest to handle correctly**: "The one type of task I've seen companies having the most problems with" (Chip Huyen) |

##### Problem Framing Changes Everything: next-app prediction (slide 17)
- **Framing 1, multiclass classification:** N classes = N installed apps, and the output is a class index. When a new app is installed you must retrain, because **adding 1 app invalidates the whole model**.
- **Framing 2, regression (scoring):** the model scores each app 0–1 as "likelihood of opening" and outputs a real-valued relevance score. When a new app is added you just score it at serving time, with **no retraining required**.

| | Multiclass | Regression |
|---|---|---|
| New app | Retrain | No retrain |
| Output | Class ID | Score |
| Ranking | Argmax | Sort by score |
| Flexibility | Low | High |

- **Key lesson:** the same real-world task can be framed as different ML tasks. **The framing determines scalability, maintainability, and cost.**
- (note: in the regression framing the app's features are part of the input, i.e. (user/context, app) → score. That is why an unseen app can be scored.)

##### Decoupling Objectives: newsfeed ranking (slide 18)
- **Goals:** filter spam, filter NSFW, filter misinformation, rank by quality, rank by engagement.
- **The naive approach is one combined loss:** **L = α · L_quality + β · L_engagement**.
  - The problem: every time you change α or β, you must **retrain the entire model**.
  - **Tuning α/β is a product decision, not an ML decision.**
- **The decoupled approach:**
  1. Train a separate spam model.
  2. Train a separate NSFW model.
  3. Train a separate quality model.
  4. Train a separate engagement model.
  5. Combine the scores at serving time: **rank = α · quality_score + β · engagement_score**.
- **The payoff:** product managers can tune α and β **without retraining any model**, so ML engineers and PMs are decoupled.

##### Quiz 2 (slides 19–20)
- **Q1:** "Our customers abandon their shopping carts too often." Frame this as an ML problem.
  **A:**
  - **Task type:** binary classification (abandon / not abandon).
  - **Inputs:** cart contents, user session history, time-on-site, device type, coupon availability.
  - **Output:** probability of cart abandonment in the next 30 min.
  - **Objective:** binary cross-entropy loss; optimise **precision at a threshold where a triggered retention action (a discount pop-up) is cost-effective**.
  - **Business metric:** **cart conversion rate** (completed checkouts / cart initiations).
- **Q2:** a content moderation system must detect hate speech and misinformation. A junior engineer proposes one model with a combined loss; a senior engineer disagrees. Why?
  **A:**
  - **The senior engineer's argument:** a combined loss requires retraining both detectors every time business priorities shift (e.g. tightening hate speech thresholds). Separate models can be **updated, A/B tested, or rolled back independently**.
  - **Separate models are better when** the tasks have different data sources, update cadences, or precision/recall requirements.
  - **One model is acceptable when** data is scarce and the tasks are highly correlated, training and serving cost is prohibitive, or the **latency budget is tight** (one inference instead of two).

#### §5 Mind vs. Data

##### The AI Hierarchy of Needs (slide 21)
- **Monica Rogati's pyramid** (the 2017 "data science hierarchy of needs" essay, widely cited), from bottom to top:
  1. Collect
  2. Move & Store
  3. Explore & Transform
  4. Aggregate & Label
  5. Learn & Optimize (AI)
- AI/ML sits at the top, and can only exist if the lower layers are solid. **"Without a data foundation, there is no data science."**
- **The great debate, mind vs. data:**
  - **Peter Norvig (Google):** "We don't have better algorithms. We just have more data."
  - **Rich Sutton, The Bitter Lesson (2019):** "The biggest lesson from 70 years of AI research is that general methods that leverage computation are ultimately the most effective."
  - **Judea Pearl:** "Data is profoundly dumb." (This is the "mind" side: causal reasoning and models matter.)

##### Data Scale Matters: The GPT Example (slide 22)
- **GPT-2 (2019)** was trained on **10 billion tokens**. **GPT-3 (2020)** was trained on **500 billion tokens**: **50× more data with the same general architecture**. The result was that qualitatively different capabilities emerged, "not from a fundamentally better algorithm — from more data and compute."
- **The practical lesson:** in many domains, collecting and cleaning more high-quality data beats spending weeks on model architecture improvements.
- **What this means for your projects:**
  - Before tuning, ask whether you need more data.
  - **Data quality often matters more than data quantity.**
  - The data collection pyramid is not glamorous, but it is what enables everything above it.
  - **"Algorithmic improvements are what win headlines. Data improvements are what win production."**
- **For this course:** we study both sides, when to invest in better algorithms and when to invest in better data. "Neither answer is always right."

Closing line: "**Frame the problem right. Everything else follows.**"

### Demo notebook (`w1s2_demo_ml_design_framework.ipynb`)
The demo has four parts: business vs. ML metrics, problem framing, decoupling objectives, and discussion. It uses numpy, pandas and sklearn (LogisticRegression, LinearRegression, train_test_split, accuracy_score, mean_squared_error).

**Part 1: Business metrics vs. ML metrics** ("Our model improved by 2% AUC — so what?")
- A toy Netflix take-rate model (illustrative numbers, not real Netflix figures):
  - 50M DAU × 10 recommendations each = **500M recs/day**.
  - Baseline take-rate is **0.28**; improved is **0.29** (+1 pp), which gives **5M extra watches/day**.
  - Assumptions: 1 extra watch = 0.0001 extra months of subscription retained, at a subscription price of **$15.49**/month.
  - `extra_monthly_revenue = extra_watches/day × 0.0001 × 15.49 × 30`, which is about $232K/month, or **about $2.8M/year (prints "~$3M/year")**.
  - Message: "That's why 'improving AUC by 0.005' needs a business translation."
- **The proxy chain:** `AUC → take-rate → watch hours → retention → revenue`. **"Each arrow is an assumption. The best ML teams measure every arrow, not just the first one."**
- **Key principle:** define the business metric first, then find an ML metric that is a reliable proxy for it. **Don't work backwards from what's easy to measure.**

**Part 2: Problem framing for next-app prediction** (a real problem solved by Google, Apple and phone OEMs)
- Synthetic data: 1000 users and 8 apps (Maps, YouTube, Gmail, Spotify, Slack, WhatsApp, Twitter, Chrome). Features are `[hour_of_day, day_of_week, last_app_id, session_length_mins]`; the label is `(last_app + hour//6) % n_apps`.
- **Framing A, classification:** `LogisticRegression` with 8 output classes. When a new app ("TikTok") launches, the model **cannot predict TikTok** and needs a full retraining cycle (hours to days).
- **Framing B, embedding similarity:** 8-dimensional app embeddings plus a user-state embedding `features @ W`. Prediction is the cosine similarity between the user embedding and each app embedding (in practice the embeddings would be trained via matrix factorization or a **two-tower** model). Adding TikTok just adds one embedding vector (from metadata/side info), so there is **no retraining and ~0 downtime**.

| | Classification | Embedding Similarity |
|---|---|---|
| New app added | Full retrain | Add one embedding row |
| 10,000 apps | 10,000-class softmax | Same model, bigger lookup table |
| Cold start | Can't predict | Use content/metadata embeddings |
| Interpretability | Class probabilities | Similarity scores |

- "**The framing decision is made before any training code is written.** It determines your system's scalability ceiling."
- (note: the slides call Framing 2 "regression (scoring)", while the notebook implements it as embedding similarity. Both are the same idea: score each candidate instead of fixing an N-class output layer.)

**Part 3: Decoupling objectives** ("Why a single loss function is a design smell")
- 2000 feed items with 6 content features.
  - The **quality** label is hard to get and requires human raters.
  - The **engagement** label is easy to get from click logs, but can be gamed.
- **Approach A, a single `LinearRegression` on a combined label** `α·quality + (1−α)·engagement` with α = 0.4. The PM then says "Users are complaining about clickbait. Reduce engagement weight" (α → 0.7). That means **retraining from scratch**, and trying 5 α values means 5 full retraining cycles. The weights are entangled, so "we can't explain what changed".
- **Approach B, separate models:** one quality model and one engagement model, combined by **runtime blending**: `final_score = α * quality_pred + (1-α) * engagement_pred`. Changing α means "change ONE number in the serving config. No retraining." Trying 5 α values costs **0 retraining cycles**.

| | Combined Model | Separate Models |
|---|---|---|
| Change business weight | Full retrain | Edit serving config |
| Debug quality issues | Hard (entangled) | Inspect quality model alone |
| Add new objective | Retrain everything | Add one new model |
| A/B test blending | Retrain per variant | Serve different configs |

- This pattern, **separate models + runtime blending**, is used for the feed rankers at **LinkedIn, Twitter and Pinterest**.
- **The price you pay:** scores must be **calibrated** across models so they are on the same scale. "But that's a much smaller problem."
- (note: there is a small bug in the demo. The retrain line computes `new_combined = alpha * quality + (1 - new_alpha) * engagement`, mixing old and new α. It doesn't change the teaching point.)

**Part 4: Discussion questions** (no single right answer; "strong answers identify tradeoffs")
1. **Metric proxy chains:** what could break AUC → take-rate → … → revenue? Give an example where improving AUC *hurts* revenue. (note: for example, the model becomes better at predicting clickbait/short-watch items, which raises clicks but lowers quality plays and satisfaction, so churn rises.)
2. **Framing:** Spotify Discover Weekly must predict songs a user will like *but has never heard*. Is that classification, regression or something else, and what is the cold-start implication for new songs? (note: a ranking/scoring or embedding-similarity framing, with content-based embeddings for new songs, as in Part 2.)
3. **Decoupling calibration:** what happens if you blend raw outputs without calibration, and how would you design a calibration approach? (note: a model with a larger output scale or variance silently dominates the blend, so α no longer means what PMs think it means. Fixes include Platt/isotonic calibration to probabilities, or normalising, e.g. z-score/percentile, before blending.)

### Pre/Post handout extras
**Pre-handout:** the session moves from *how* ML systems are built to *why* they are built that way, with every decision grounded in DMLS Ch. 2. Planned items: an opening quiz ("what business metric does your favourite ML use case move?"), the translation table, the Netflix chain, the four requirements, the 6-step cycle, a framing exercise, a live newsfeed decoupling example, and the Norvig/Sutton/Pearl debate. Preparation: think of a project and its business metric, recall the 5-question framework, and optionally read DMLS Ch. 2 ("Framing ML Problems") and Sutton's *The Bitter Lesson* (a 5-minute read at incompleteideas.net).

**Post-handout key takeaways:** (1) Business metrics first, always: what should move, by how much, and how the ML objective traces back to it. (2) The four requirements are "structural load-bearing requirements", not optional features. (3) The process is iterative and never-ending: "Design systems that expect change, not systems that assume stability." (4) Problem framing changes everything: whether a new class needs retraining, whether PMs can tune weights without ML engineers, and maintainability at scale. "Spend time on framing before spending time on models."

**Test Your Understanding (with handout answers):**
- **Q1.** A music recommender's offline NDCG stays high, but session length drops 12% after 6 weeks. What went wrong?
  **A:** NDCG measures ranking quality on held-out data, not whether users *acted*. The likely causes are that the model optimised for historical click patterns that no longer reflect current preferences (**concept drift**), or that the recommendation set **lacked diversity**. The team should measure a business-aligned proxy instead: session length, **weekly active listening days**, or a take-rate equivalent (**plays per recommendation shown**).
- **Q2.** Why not have one model that jointly predicts spam, hate speech, misinformation and NSFW?
  **A:** With a combined loss, changing any weight requires **full retraining**, which is expensive and slow and ties product decisions to ML engineering cycles. Instead, use separate detectors whose scores are combined at serving time with tunable weights. PMs adjust the weights, and engineers retrain individual models independently.
- **Q3.** Frame 30-day churn.
  **A:** Binary classification. Inputs (at least 3): days since last purchase, total spend in the past 90 days, number of sessions in the past 30 days. Output: P(churn within 30 days). Objective: binary cross-entropy. Business metric: **30-day customer retention rate**. Churn probability should inversely correlate with observed retention, so **validate this empirically before and after deployment**.
- **Q4.** Reliability vs. adaptability?
  **A:** Reliability means correct predictions despite adversity (bad inputs, infrastructure failures). Adaptability means evolving with shifting distributions without interruption.
  - **High reliability with low adaptability:** a fraud model running 24/7 without errors whose accuracy declines to **60% over 18 months** as fraud patterns shift. It never crashes, but it stops being useful.
  - **High adaptability with low reliability:** a system that retrains daily but has **no input validation**, so bad data periodically causes silent garbage predictions.

**Design exercises:**
1. **Cart abandonment** ("reduce by 15% this quarter"): (a) give the full framing with at least 4 input features and a named loss; (b) say whether it is regression or classification, whether it could be framed either way, and the trade-offs; (c) name the business metric to track and how it connects to the ML metric; (d) identify one reliability concern and one adaptability concern.
2. **Newsfeed decoupling** (spam filter + quality rank + engagement rank, currently one weighted loss): (a) describe the decoupled architecture (how many models, what each predicts, how they combine); (b) a PM wants more quality weight for a weekend experiment: what does that take in the coupled vs. the decoupled design (retrain vs. config change); (c) give a scenario where coupling is preferable (hint: data scarcity or latency constraints).
- **Challenge:** pick Spotify, Uber or LinkedIn and name 3 ML use cases. For each, give (1) a *specific* business metric (e.g. "7-day retention rate", not "engagement"), (2) the framing (task type, input, output, objective), (3) which of the four requirements is most critical and why, and (4) one silent-failure mode and how to detect it. About one page; "precision and specificity are more valuable than length."

**Further reading:** Chip Huyen DMLS Ch. 2 (sections on Framing ML Problems, Decoupling Objectives, Mind vs. Data); Sutton *The Bitter Lesson* (2019); Rogati *The AI Hierarchy of Needs* (2017, hackernoon); Norvig et al. *The Unreasonable Effectiveness of Data* (2009, IEEE Intelligent Systems vol. 24, the source of the "more data beats better algorithms" idea). Online: huyenchip.com, eugeneyan.com, the Netflix Tech Blog (take-rate), RecSys papers (ranking and objective decoupling), The Batch, and the Evidently AI blog (adaptability and monitoring).

### Likely doubts & exam angles
1. **Why do ML projects get killed?** Teams optimise ML metrics (AUC, F1) that don't trace to a business metric, so the work looks like "6 months and nothing changed". Before building, ask what business metric moves, by how much, in what timeframe, how the ML metric connects, and what being wrong costs.
2. **Define take-rate and its chain.** Take-rate = # quality plays / # recommendations shown. Better recommender → higher take-rate → more streaming hours → lower cancellation.
3. **The four requirements, and one failure example for each?** Reliability: silent wrong predictions (the Google Translate problem). Scalability: Prime Day at $72–99M/hr, 8K models. Maintainability: only the original author can change the system. Adaptability: the model goes stale and needs a rebuild.
4. **Reliability vs. adaptability?** Reliability is correctness under adversity *now*; adaptability is evolving over time *without downtime*. You can have one without the other (the fraud model at 60% after 18 months).
5. **The 6 steps, in order?** Scoping → Data Eng → Model Dev → Deployment → Monitoring & Continual Learning → Business Analysis → (repeat). The cycle is non-linear and never-ending.
6. **High AUC but the business metric is flat?** The model may be fine and the *intervention* broken. Check whether the predictions triggered an action, whether the action worked, and whether the metric and threshold are right (precision at a cost-effective point).
7. **The three parts of an ML problem?** Inputs, outputs and the objective function. "Slow support" becomes multiclass routing to 4 departments, minimising the misrouting rate.
8. **Which task type is hardest, and what is the data rule of thumb?** Multilabel is hardest. Multiclass needs about 100K examples for 1,000 classes.
9. **Classification vs. regression framing for next-app prediction?** Classification needs a retrain for every new app; scoring/regression (or embedding similarity) just scores the new app at serving time.
10. **When should you couple objectives into one model?** When data is scarce and the tasks are highly correlated, when training/serving cost is prohibitive, or when the latency budget is tight. Otherwise decouple, and remember that decoupling needs score calibration.
11. **Mind vs. data positions?** Norvig: more data beats better algorithms. Sutton: general methods that leverage computation win. Pearl: "data is profoundly dumb". Rogati: no AI without the data foundation. GPT-2 → GPT-3 was 10B → 500B tokens (50×) with the same architecture.
12. **Deployment maturity numbers?** 75% of mature (5+ years) companies deploy a new model in < 30 days; 60% of beginners take > 30 days for their first.

---

## Cross-links
- **The 5-Question Framework (1.1)** is the course-wide template for every system designed later, including the case-study weeks 5–8. Session 1.2 deepens Q1 (problem, metrics, framing).
- **Feature store, online/offline, point-in-time correctness, training-serving skew** lead into Week 2 Session 1 (feature pipelines).
- **Serving patterns, latency budgets (p50/p95/p99), sync/async/streaming/edge, caching, quantisation and distillation** lead into Week 2 Session 2 (model serving) and Week 3 Session 1 (LLM serving).
- **Model registry, experiment tracking (MLflow, "used from Week 3"), retraining triggers, canary/shadow deploys, drift monitoring and the alerting hierarchy** lead into Week 3 Session 2 (MLOps).
- **Eugene Yan's LLM patterns** are flagged as "relevant from Week 4" (RAG pipelines, agents).
- **Decoupled objectives with runtime blending and calibration** recur in recommendation/ranking system designs (feeds, recsys case studies).
- **Data vs. algorithms (GPT-2 → GPT-3 scaling)** frames the LLMOps half of the course.

## Glossary
- **ML Systems Engineering**: everything between a working notebook and a reliable production system.
- **Six components**: Data Collection, Feature Store, Training Pipeline, Model Registry, Serving Layer, Monitoring (plus a feedback/retrain loop).
- **Training-serving skew**: features at serving time differ from those at training time (e.g. an unsaved scaler), causing a silent accuracy drop.
- **Silent degradation**: model quality decays with no errors logged.
- **Data drift**: a shift in the input distribution P(X).
- **Concept drift**: a shift in P(Y|X), the input→label relationship.
- **Label shift**: a shift in the output distribution P(Y).
- **Feedback loop / exposure bias / closed-loop amplification**: predictions become future training data and reinforce bias.
- **Glue code**: custom wrapper code around generic models that ends up carrying the real logic.
- **Pipeline jungle**: duplicated, unowned, untraceable data pipelines.
- **Dead experimental codepaths**: unused experiment code kept "in case".
- **Undeclared consumers**: systems silently reading another model's or table's outputs with no contract.
- **Correction cascade**: stacking models to patch the errors of upstream models (A → B → C).
- **Entanglement**: data dependencies where changing one input changes everything.
- **Offline feature store**: historical, columnar (Parquet/Delta) feature values for training, with point-in-time correctness.
- **Online feature store**: low-latency KV (Redis/DynamoDB) feature lookup at inference.
- **Point-in-time correctness**: a feature value at time t uses only data available before t, which prevents leakage.
- **Staleness budget**: the maximum acceptable age of an online feature value.
- **Model registry**: a versioned catalogue of models with lineage, metrics, lifecycle states (Staging → Production → Archived), an audit log and rollback.
- **Latency SLA / p99**: the guaranteed tail-latency bound. Budget = feature lookup + preprocessing + inference + postprocessing.
- **Pareto frontier (latency vs. accuracy)**: the set of models with no option that is both faster and more accurate.
- **Retraining triggers**: scheduled, drift-triggered or performance-triggered.
- **Alerting hierarchy**: data quality → input drift → output drift → business metric.
- **5-Question Framework**: Problem → Data → Model → Serve → Monitor.
- **Take-rate**: # quality plays / # recommendations shown (Netflix).
- **Proxy ML metric**: an ML metric that should move with the business metric; verify this empirically.
- **Proxy chain**: AUC → take-rate → watch hours → retention → revenue. Each arrow is an assumption.
- **Deployment maturity curve**: mature organisations ship models much faster (75% in < 30 days vs. 60% of beginners taking > 30 days).
- **Reliability**: correct behaviour despite bad inputs, infrastructure failures and edge cases.
- **Scalability**: growth along three axes: traffic, complexity and model count.
- **Maintainability**: multiple teams can understand and change the system; versioned data; reproducible models.
- **Adaptability**: evolving with data and business shifts without service interruption.
- **Canary deployment / shadow mode**: gradual or silent rollout of a new model before full launch.
- **Feature flags**: infrastructure for quick rollbacks or toggles.
- **6-step cycle**: Scoping → Data Engineering → Model Development → Deployment → Monitoring & Continual Learning → Business Analysis.
- **ML problem**: inputs + outputs + objective function.
- **Regression / binary / multiclass / multilabel**: the task-type taxonomy (losses: RMSE/MAE, BCE, CE; multilabel is the hardest).
- **Decoupling objectives**: separate models per objective, blended at serving time with tunable weights (α, β).
- **Calibration**: putting scores from different models on a comparable scale before blending.
- **Two-tower / embedding similarity**: user and item embeddings scored by similarity; new items are added as new embedding rows.
- **Cold start**: predicting for a new item or user with no history; handled through content/metadata embeddings.
- **AI Hierarchy of Needs (Rogati)**: Collect → Move & Store → Explore & Transform → Aggregate & Label → Learn & Optimize.
- **The Bitter Lesson (Sutton, 2019)**: general methods that leverage computation win in the long run.
