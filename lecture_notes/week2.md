# Week 2 — The Data Layer & The Serving Layer (Feature Stores, Data Pipelines, Model Serving)

Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week2/`):
- Session 2.1: `w2s1_slides.pdf`, `w2s1_pre_handout.pdf`, `w2s1_post_handout.pdf`, `w2s1_demo_feature_pipelines.ipynb`
- Session 2.2: `w2s2_slides.pdf`, `w2s2_pre_handout.pdf`, `w2s2_post_handout.pdf`, `w2s2_demo_model_serving.ipynb`

Course: ML System Design and LLMOps — Priyansh Saxena, Scaler School of Technology. Each session ~2 hours.
Primary text referenced all week: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly 2022) — Ch. 3 (Data Engineering Fundamentals) and Ch. 5 (Feature Engineering) for 2.1; Ch. 7 (Model Deployment and Prediction Service) and Ch. 1 (latency/throughput) for 2.2.

---

## Session 2.1 — Feature Stores, Data Pipelines & Offline/Online Serving

Subtitle: "Data Sources • Formats • Feature Engineering • Feature Stores". Prerequisite: Week 1 Session 1 (ML Systems 101).
Framing (pre-handout): last session = big picture (six components of a production ML system, many failure modes); today = the **foundation: data**. "Every model is only as good as the data it trains on and the features it serves with." Demo notebook opener: "A bad model with great data will outperform a great model with bad data, every time."

Agenda (5 sections):
1. Data Sources and Formats
2. Data Models and Storage
3. Batch vs. Stream Processing
4. Feature Engineering
5. Feature Stores

Closing slide: "Bad features in, bad predictions out. Garbage in, garbage out — forever."

### TL;DR
- **Format matters**: Parquet (column-major, binary) is ~**6× smaller** and **2× faster** to unload/read than CSV on S3. Rule: **writes ⇒ row-major (CSV, DBs); feature reads ⇒ column-major (Parquet)**.
- **Storage/access patterns**: OLTP (writes, point lookups, row-oriented) vs. OLAP (aggregate reads, column-oriented). Typical flow: raw events → OLTP → nightly ETL → OLAP → feature pipelines → Parquet in S3. Redis/key-value = the online feature store.
- **Batch vs. stream**: the key question is "**how fresh does the feature need to be?**" (fraud: seconds ⇒ streaming; weekly forecast ⇒ batch).
- **Leakage is silent and deadly**: **split first, then compute feature statistics on the training split only**; respect **point-in-time correctness** (for a training row at time t, only use data < t); use time-based splits for temporal data.
- **Feature store = offline store (training, historical, S3+Parquet, point-in-time joins) + online store (serving, Redis/DynamoDB, sub-ms)**, with "**compute once, use everywhere**" ⇒ no training-serving skew. "A feature store is not just storage — it is a **contract** between data engineering and model serving."

### Concepts (in lecture order)

#### §1 Data Sources and Formats

##### 1.1 Where does ML data come from? (four source types)
| Source | Characteristics / challenges |
|---|---|
| **User input data** | Text, images, videos, uploaded files. "**If users can input wrong data, they will.**" Requires heavy validation at ingest. Users expect immediate results ⇒ fast processing needed. |
| **System-generated data** | Logs: system state, events, model predictions. "**Log everything**" — "interesting" in production usually means a crash or an astronomical cloud bill. Immutable audit trail for debugging. |
| **Internal databases** | Company assets, CRM records, inventory. Typically structured, well-maintained. Access via SQL or internal APIs. |
| **Third-party data** | First-party: your own users; Second-party: another company's customers; Third-party: aggregated from public sources. Example: **Apple IDFA change (2021)** made tracking opt-in ⇒ "devastated the **$80B** ad industry overnight" (risk of depending on third-party data). |

##### 1.2 Data formats: JSON vs. CSV vs. Parquet
| | JSON | CSV | Parquet |
|---|---|---|---|
| Storage layout | Key-value text | Row-major text | Column-major binary |
| Readability | Human-readable | Human-readable | Binary (not readable) |
| File size | Large (verbose) | Medium | Small (**6× less than CSV on S3**) |
| Best for | APIs, configs | Writing rows | Reading features |
| Speed | Slow to parse | Slow column reads | **2× faster to unload vs. CSV** |
| Tools | Ubiquitous | pandas, Excel | Spark, pandas, DuckDB |

**Why Parquet dominates ML pipelines**: 6× less storage and 2× faster reads than CSV on S3. Column-major layout means you can read only the columns you need without scanning entire rows.

##### 1.3 Row-major vs. column-major — why it matters
- **Row-major (CSV, JSON)**: rows stored contiguously on disk. Fast to write one sample at a time; fast to retrieve all features of one sample; **slow to read a single feature across all samples**. Good for: transactional writes, streaming inserts.
- **Column-major (Parquet, ORC)**: columns stored contiguously. Fast to read one feature across all samples; **excellent compression** (same-type data clusters together). Good for: analytics, ML feature reads.
- **The pandas consequence**: pandas is built around column-major format internally. Iterating a DataFrame **by row: 2.41 s**; **by column: 0.07 s** ⇒ **~34× slower by row**. Always prefer vectorised column operations.
- **Rule of thumb**: **Writes ⇒ row-major (CSV, databases). Feature reads ⇒ column-major (Parquet).**

##### 1.4 Structured vs. unstructured data
| Structured | Unstructured |
|---|---|
| Well-defined schema: columns, types, constraints | No predefined schema |
| Lives in relational DBs or data warehouses | Text, images, audio, video, sensor readings |
| Easy to query with SQL | Requires ML to extract signal |
| e.g., user age, order amount, transaction count | Typically **80–90%** of all industry data |
| Typically **10–20%** of enterprise data | Stored in object stores (S3, GCS) or data lakes |

Industry reality: "In industry, 80–90% of data is unstructured." — Chip Huyen. Most of an ML engineer's time is spent making unstructured data usable for models.

##### 1.5 Quiz 1 (with answers)
- **Q1 Format selection**:
  - (a) Stream 10,000 click events/sec from web app to data lake → **row-major / JSON**: each event is a single row written immediately; low-latency ingestion matters more than read speed.
  - (b) Nightly batch query: average session duration over 100M user rows → **Parquet (column-major)**: scanning one column across 100M rows is 6× faster and cheaper than CSV.
  - (c) REST API returning one user's profile → **JSON**: human-readable, schema-flexible, universally supported by API clients.
- **Q2 Unstructured data** (sentiment classifier on support-chat .txt files in S3) — two data quality problems:
  - **Language/encoding issues**: mixed encodings (UTF-8, Latin-1) or languages; an English-trained model silently fails on non-English transcripts. Fix: detect/filter by language; normalise encoding at ingest.
  - **Label imbalance / missing labels**: if 90% are negative, always predicting "negative" gives 90% accuracy. Fix: audit label distribution; oversample or reweight minority classes.

#### §2 Data Models and Storage

##### 2.1 The relational model: tables and normalization
- Data organised into **relations (tables)** of **tuples (rows)**; each tuple has a unique key; relations linked via **foreign keys**.
- **Normalization** reduces redundancy: 1NF, 2NF, 3NF. Query via **SQL — declarative, set-based**.
- **Harry Potter publisher example**: without normalization the publisher name is stored in every book row ⇒ publisher changes name ⇒ update 4 rows, easy to miss one. With normalization: one Publishers table, books reference by ID ⇒ update one row, consistent everywhere.
- **When to use relational DBs**: structured data with clear schema; transactions requiring **ACID** guarantees; complex joins across entities (user accounts, orders, inventory).
- **ML relevance**: most production ML systems have a relational backbone for structured metadata (user profiles, item catalogs, label stores). Features are derived from these tables.

##### 2.2 NoSQL: four models for different problems
| Type | Structure | Tools | Use in ML |
|---|---|---|---|
| Document | JSON-like documents; flexible schema | MongoDB, CouchDB | User profiles, content metadata |
| Key-value | Hash map; ultra-low latency | Redis, DynamoDB | **Online feature store**; session cache |
| Graph | Nodes and edges; relationship queries | Neo4j, Neptune | Fraud graph features; social network |
| Column-family | Wide rows; sparse columns | Cassandra, HBase | Time-series features; event streams |

**The online feature store pattern**: the most common NoSQL use in ML is **Redis as an online feature store**. Pre-computed features are written to Redis (slide: "at training time") and read at **sub-millisecond latency** at serving time.

##### 2.3 OLTP vs. OLAP
| OLTP — Online Transaction Processing | OLAP — Online Analytical Processing |
|---|---|
| Optimised for **writes and point lookups** | Optimised for **reads and aggregate queries** |
| Low latency, high concurrency | Large scans across millions of rows |
| Small transactions: one user, one order | "How many orders in the last month?" |
| e.g., user signs up, order placed, payment processed | e.g., business reporting, feature aggregation, model evaluation |
| **Row-oriented** storage | **Column-oriented** storage |
| PostgreSQL, MySQL, Aurora | Redshift, BigQuery, Snowflake, ClickHouse |

**ML data engineering in practice**: Raw events land in OLTP → nightly ETL moves data to OLAP → feature engineering pipelines run against OLAP → training data materialized to **Parquet in S3**.

##### 2.4 Moving data between services (three modes of data passing)
| Via databases | Via services (REST/gRPC) | Via real-time transport (Kafka) |
|---|---|---|
| Service A writes to a shared DB; Service B reads from it | A sends HTTP request to B's API | Event-based, **pub/sub** model |
| Simple, auditable | RESTful APIs: JSON over HTTP | Producers write to **topics**; consumers read |
| Adds latency; poor for real-time | **gRPC: Protocol Buffers, binary, faster** | **Data broker decouples producers from consumers** |
| Tight coupling on schema | **Synchronous**: caller waits | Enables stream processing (Flink, Spark Streaming) |
| | Good for request-response patterns | Best for high-throughput, event-driven ML |

Diagram: Producer A → Kafka Topic → (decoupled) → Consumer B, Consumer C (one producer, many independent consumers).

Break slide: "You just stored data in three formats, four databases, and two processing paradigms."

#### §3 Batch vs. Stream Processing
| Batch processing | Stream processing |
|---|---|
| Periodically process large accumulated data | Process data as each event arrives |
| High throughput; high latency (minutes to hours) | Low latency (ms to seconds); lower throughput |
| Tools: MapReduce, Apache Spark | Tools: Apache Flink, Spark Streaming, Kafka Streams |
| Batch-processed data = **historical data** | Real-time processed data = **streaming data** |
| e.g., nightly training data aggregation, weekly retrain | e.g., real-time fraud detection, live recommendation refresh |
| Cheaper; simpler to operate | More complex; **stateful aggregations are hard** |

**The key question: how fresh does the feature need to be?** Fraud detection: features must reflect the last few seconds ⇒ streaming. Weekly demand forecast: historical aggregates fine ⇒ batch.
Post-handout: "Knowing which of your features need streaming is a key design decision — streaming adds significant infrastructure complexity."

#### §4 Feature Engineering

##### 4.1 Why features matter more than algorithms — the Facebook finding (2014)
- Facebook ad click prediction (CTR) paper at Facebook scale: **top 10 features ≈ 50%** of the model's total feature importance; **the last 300 features contributed < 1%**.
- Implication: feature selection matters as much as model choice. Quote: "Having the right features is the most important thing in developing their ML models."
- Common operations: handling missing values; scaling and normalization; encoding categoricals; feature crossing.

##### 4.2 Handling missing values
- **Deletion** (drop rows or columns with nulls). Risk: introduces **selection bias** if data is not **MCAR** (missing completely at random).
- **Imputation** (mean, median, mode, or model-predicted value). Risk: injects noise if imputation is wrong. **Always impute on training stats only.**
- "Neither is universally correct — understand **why** values are missing."

##### 4.3 Scaling
- **Normalization (min-max) to [0,1]**: x' = (x − x_min) / (x_max − x_min). Use when: bounded range, no outliers.
- **Standardization (μ=0, σ=1)**: x' = (x − μ) / σ. Use when: Gaussian-ish, outliers present.
- **Critical: fit on training data only.** Compute mean/std/min/max on the training split; apply those stats to val and test. **Fitting on the full dataset is data leakage.**

##### 4.4 Feature crossing
- Combine two or more features to capture interactions. Example: **marital status × number of children** — better predictor of house-buying intent than either alone.
- Common in logistic regression and tree ensembles. Deep learning learns crosses implicitly; explicit crosses help shallow models.
- Risk: **exponential feature explosion** if unconstrained.

##### 4.5 Data leakage — "the silent accuracy inflation trap"
Your validation accuracy is suspiciously high. Did you:
- Scale before splitting train/test?
- Use future information in a feature?
- Include the label in a proxy feature?

**Rule: always split first, then compute feature statistics.** Quote: "It's hard avoiding data leakage without understanding how your data is collected." — Chip Huyen.
Demo-notebook leakage variants (all same symptom — offline great, online disappoints):
- Leakage through **features** (feature computed using future events)
- Leakage through the **target** (label derived from data that wouldn't exist at prediction time)
- **Group leakage** (same user in train and test — you've seen their behavior pattern)

##### 4.6 Point-in-time correctness
- Problem: building a training row for a recommendation event on date **D**, feature = "user's total purchases in the past 30 days".
  - **Wrong**: compute using all available data up to today (includes future).
  - **Correct**: compute using only data available **before D**.
- Diagram: timeline with purchases $50, $30 (before D — "use these") and $80 (after D — "leakage: includes future"); training event at D.
- Why it matters: violation = data leakage using future info; model appears to work perfectly offline; **fails completely in production where future is unavailable**.
- **Feature stores perform point-in-time joins**: for each training row at time t, return only features computed from data **< t**. Feature store enforces this automatically.

#### §5 Feature Stores

##### 5.1 Offline + online architecture
```
            Raw Data (S3 / Kafka)
              /              \
   Offline Store  --sync-->  Online Store
   (S3 + Parquet)            (Redis)
        |                        |
   Training Pipeline        Serving (Inference)
```
- **Offline store (training)**: historical feature values for model training; storage **S3 + Parquet / Delta Lake**; **point-in-time correct** retrieval; prevents training-serving skew; batch compute: **Spark, dbt**.
- **Online store (serving)**: pre-computed features for real-time inference; storage **Redis, DynamoDB** (key-value); **sub-millisecond reads**; must stay in sync with offline store; **staleness budget**: how stale is acceptable?
- **Compute once, use everywhere**: feature logic defined once; training and serving use identical computations ⇒ **no training-serving skew**.

##### 5.2 Why separate offline and online?
- **Problem without separation**: training reads features from S3 batch job (cheap, slow); serving recomputes features from DB on every request (expensive, fragile) ⇒ **different feature values at train vs. serve time**; debugging = impossible to reproduce a production prediction.
- **Problem with only a database**: production DB cannot sustain training reads (millions of rows); serving queries slow (complex aggregations); **no historical versioning** — "what were the features on Jan 1st?"
- **What separation gives you** (5 benefits):
  1. Training: point-in-time correct historical features
  2. Serving: sub-millisecond reads for online inference
  3. Consistency: same feature definitions, same values
  4. Reuse: team A's feature reusable by team B
  5. Auditability: reproduce any prediction from any date
- Tools: **Feast** (open source), **Tecton** (managed), **Hopsworks** (managed/open).

##### 5.3 Feature store tools
| Tool | Type | Offline store | Online store |
|---|---|---|---|
| Feast | Open source | S3, GCS, Redshift, BigQuery | Redis, DynamoDB, SQLite |
| Tecton | Fully managed (SaaS) | S3 + Delta Lake | DynamoDB, Redis |
| Hopsworks | Managed / open source | S3, HDFS | RonDB (MySQL Cluster) |
| Vertex AI FS | GCP managed | BigQuery | Bigtable |
| SageMaker FS | AWS managed | S3 + Glue | DynamoDB |

Demo teaser (Feast): define a feature view; materialize batch features to the offline store; point-in-time join for training dataset generation; online feature retrieval for serving. (note: the actual notebook implements a pure-Python simulation rather than Feast — see below.)
**Key insight**: "A feature store is not just storage — it is a **contract between data engineering and model serving**. It defines what a feature means and guarantees consistency."

##### 5.4 Quiz 2 (with answers)
- **Q1 Point-in-time** (churn model; feature "total support tickets submitted"; colleague uses all tickets up to today): the training row for date D includes tickets submitted after D ⇒ **future information leakage**; model learns "this user will submit 5 more tickets before churning", impossible to know at serving. **Fix**: point-in-time join returns feature computed only from data with timestamp < D.
- **Q2 Batch vs. streaming** (real-time fraud):
  - (a) Total spend last 30 days → **batch**: nightly Spark job writes to offline/online store. Infra: **data warehouse + Spark + feature store**.
  - (b) Transactions in last 5 minutes from this device → **streaming**: window too short for batch; **Kafka + Flink (or Spark Streaming)** maintaining a rolling 5-min counter per device. Infra: **Kafka + Flink + Redis (online store)**.

##### 5.5 Recap slide ("What we covered")
§1: four data source types; JSON vs CSV vs Parquet; row vs column-major; 80–90% unstructured. §2–3: relational/normalization; four NoSQL types; OLTP vs OLAP; three ways to pass data. §4: features > algorithms (Facebook); deletion vs imputation; scaling fit on train only; split first then scale; point-in-time. §5: offline store (training, historical, Parquet/S3), online store (serving, sub-ms, Redis), compute once use everywhere, Feast/Tecton/Hopsworks.

### Demo notebook — `w2s1_demo_feature_pipelines.ipynb` (instructor-run)
Four parts: (1) data formats, (2) the leakage trap, (3) point-in-time feature construction, (4) a simple feature store simulation.

**Part 1 — CSV vs Parquet.** Generates 500,000-row × 10-col transaction dataset (`user_id, item_id, timestamp, price, quantity, category, user_age, session_length, device_type, converted`), writes `df.to_csv` and `df.to_parquet`, compares file size and best-of-3 read time (`time.perf_counter`), then a column-selective read:
```python
pd.read_parquet(parquet_path, columns=["user_id", "converted"])   # reads only 2 columns from disk
pd.read_csv(csv_path, usecols=["user_id", "converted"])           # still parses every row
```
Takeaways: at **10M rows CSV ~8 s vs Parquet ~1 s** per feature engineering job; at **1B rows** you can't fit CSV in memory — Parquet lets you read only needed columns. **Rule of thumb: "Use CSV for files you're going to look at. Use Parquet for files your code will read."**

**Part 2 — The leakage trap ("reproduce that failure in 20 lines").** 10,000 hourly observations (2023-01-01 → 2023-04-10), 5 features with a time trend (`page_views_7d`, `purchases_7d` trending up; `avg_session_length`, `items_in_cart` stable; `recency_score` trending = leakage source); label = will_convert_next_day via logistic of trend features. `GradientBoostingClassifier(n_estimators=50)`.
- Split A: `train_test_split(test_size=0.2)` random (WRONG for time series).
- Split B: time-based, first 70% train / last 30% test (CORRECT).
- Prints AUC/accuracy for both and the "inflation from leakage" gap. Explanation: with random split the test set contains future data used in training; the model "memorised" patterns from the future; in production there's no future data.
- **Rule: "For any ML problem involving time: your test set must only contain data from after your training cutoff."** "The rule is simple, the discipline is hard."

**Part 3 — Point-in-time feature construction.** 200-event transaction log for 10 users (purchase/view/add_to_cart), three prediction events (users 3, 7, 2 at 2023-01-10/15/20). Core function:
```python
def build_point_in_time_features(user_id, cutoff_time, txn_log, lookback_days=30):
    window_start = cutoff_time - pd.Timedelta(days=lookback_days)
    user_txns = txn_log[(txn_log.user_id == user_id) &
                        (txn_log.event_time <  cutoff_time) &   # point-in-time filter
                        (txn_log.event_time >= window_start) &
                        (txn_log.event_type == "purchase")]
    # -> tx_count_30d, avg_amount, total_spend, days_since_last (999 if none)
```
Then the WRONG way: join on `user_id` without the time filter ⇒ more transactions counted, "includes FUTURE data!" (prints "should be 3"). Pitfall: "your feature pipeline might join on `user_id` without filtering by timestamp."
**Why feature stores exist**: ensures (1) the exact same function is used at training (offline) and serving (online); (2) the time filter is enforced automatically; (3) features are cached so you don't recompute every training run. "**The simplest feature store is just: a function + a consistent calling convention.**"

**Part 4 — Mini feature store simulation.** Raw events dict for `user_1..user_5` (date, spend). One shared function `compute_user_features(user_id, cutoff_time, events)` → `tx_count_30d, avg_spend, active_days, spend_velocity (= sum(spend)/30)` over a 30-day window `[cutoff-30d, cutoff)`.
- Offline store: dict (in prod: **Parquet on S3**, filled by a Spark/Airflow batch job) at `training_cutoff = "2023-06-01"`.
- Online store: dict (in prod: **Redis**), filled by the **same function** run more frequently (e.g., hourly).
- Consistency check: `offline_store["user_3"] == online_store["user_3"]` → `True` — "training-serving consistency guaranteed". Key insight: **computation is shared; only storage differs.**
- **What a production feature store adds**: **Versioning** (each feature definition versioned; old models use old definitions), **Backfilling** (compute features for any historical date on demand), **Monitoring** (detect offline vs online distribution drift), **Discovery** (catalog for reuse). Tools: Feast, Tecton, Hopsworks, Vertex AI Feature Store.
- **Discussion question**: online store pre-computed at midnight; a user makes 5 purchases at 2am — what's the staleness trade-off and how to mitigate? (note: expected direction — staleness budget per feature; increase materialization frequency or add a streaming update path (Kafka+Flink → Redis) for features where freshness matters.)

### Pre/Post handout extras
**Pre-handout — four themes**: Data as the Foundation; Pipelines That Move Data (Kafka as data broker: what it decouples and why); Feature Engineering Traps (leakage from scaling before splitting; future info in training rows); Feature Stores Offline & Online.
**Session plan**: opening discussion "where do your features come from right now?"; format comparison (hands-on row vs column-major); data leakage demo; feature store architecture walkthrough; two quizzes; Feast demo.
**Prep**: recall your last ML project — were features computed the same way at training and serving time? Think: how would you compute "user's average session length" during training on historical data vs. during serving on a live request? Optional: Chip Huyen Ch. 3; Feast quickstart (feast.dev); Google Colab access.

**Post-handout key takeaways** (4): data format choice affects performance; leakage is silent and deadly (split first, then compute stats on training split only); feature stores solve training-serving skew ("the same feature computed differently in two places — a silent but catastrophic bug"); streaming enables real-time ML (but adds infra complexity).

**Test Your Understanding (with handout answers)**:
- Q1: Spark job over **500GB** on S3, colleague wants CSV intermediates. Two reasons for Parquet: (1) **Storage** — columnar compression ~6× smaller ⇒ 500GB → ~**80GB**, cutting S3 cost and network I/O; (2) **Speed** — **predicate pushdown and column pruning**: Spark reads only needed columns vs CSV full scan; for queries touching 5–10 of 100 columns ⇒ **10–20× speedup**.
- Q2: churn model **91% AUC validation → 67% deployed**; data sorted by date, random 80/20 split. Answer: **temporal data leakage** — random split mixed time periods, time-based features used future data. Fix: sort by date, split at a cutoff (train before, validate after).
- Q3: OLTP vs OLAP for ride-sharing: OLTP (PostgreSQL) — recording a ride booking in real time when the user confirms a trip; OLAP (BigQuery) — average fare per city over past 30 days to feed a pricing model.
- Q4: "number of videos watched in the last 1 hour" for recommendations — **Both**: offline holds historical versions for point-in-time correct training; online holds current value updated by a streaming job for sub-ms serving. "**Offline = training correctness; online = serving freshness.**"

**Design exercises**:
- Ex 1: Feature pipeline for a ride-sharing **ETA** model — ≥6 features, classify each batch vs streaming; sources (OLTP / Kafka / external API); which need cross-source joins.
- Ex 2: 2 years of transactions with `transaction_date`, colleague proposes random **70/15/15** split — explain leakage for time-based features (e.g., purchases past 7 days); describe correct time split and date boundaries; what the validation set measures that training can't (note: generalisation to future, unseen time periods).
- **Challenge**: data stack for fintech fraud detection, decision within **200 ms** of every transaction: (1) data sources (user events, bank APIs, device signals; batch or stream), (2) storage formats/systems (raw landing, format, OLTP/OLAP), (3) 5 features with batch/streaming + **staleness budget**, (4) feature store design (offline vs online, sync, point-in-time strategy), (5) serving: retrieval at inference + **latency breakdown** (feature lookup, model inference, response). One paragraph each, justify every choice.

**Further reading**: Chip Huyen DMLS Ch. 3 & 5; Feast docs (quickstart 15 min, "highly recommended before Week 2 Session 2 lab"); Martin Kleppmann *Designing Data-Intensive Applications* Ch. 3 (& 4) — storage engines, row vs column, OLTP vs OLAP, encoding; Maxime Beauchemin *Functional Data Engineering* (2018) — idempotent pipelines, immutable data, why point-in-time correctness matters, pipelines that produce the same output every run. Online: Feast, DuckDB (in-process OLAP on Parquet, great for local feature dev), Tecton blog, Kafka docs, Hopsworks blog, Chip Huyen's blog.

### Likely doubts & exam angles
1. **Why is Parquet faster for ML?** Column-major: read only needed columns (column pruning), predicate pushdown, same-type data compresses well ⇒ ~6× smaller, ~2× faster than CSV on S3.
2. **When would I still use row-major/JSON?** Writes, streaming inserts (10K events/sec ingestion), API payloads, configs; when retrieving all fields of one record.
3. **Why does iterating a DataFrame by row hurt?** pandas is column-major internally: 2.41 s by row vs 0.07 s by column (~34×). Vectorise.
4. **OLTP vs OLAP in one line?** OLTP = row-oriented, writes/point lookups, ACID (Postgres); OLAP = column-oriented, big aggregate scans (BigQuery/Snowflake). Features are computed against OLAP.
5. **REST vs gRPC vs Kafka?** REST = JSON over HTTP, synchronous; gRPC = Protocol Buffers, binary, faster, synchronous; Kafka = async pub/sub broker that decouples producers from consumers, enables stream processing.
6. **Is min-max scaling on the full dataset really leakage?** Yes — val/test statistics leak into training. Split first, fit scaler on train, apply to val/test. Same for imputation.
7. **Random split on time-sorted data — what goes wrong?** Temporal leakage: training sees future; offline metrics inflate (91%→67% AUC example). Use a time cutoff split.
8. **What is a point-in-time join?** For each label row at time t, join the feature value computed only from data with timestamp < t.
9. **Offline vs online store?** Offline = historical, S3+Parquet/Delta, batch (Spark/dbt), point-in-time training data. Online = latest values, Redis/DynamoDB, sub-ms lookups for serving. Same feature definitions ⇒ no training-serving skew.
10. **Batch or streaming for a given feature?** Ask "how fresh must it be?" Long windows (30-day spend) ⇒ batch (warehouse+Spark). Short windows (5-min count) ⇒ streaming (Kafka+Flink+Redis).
11. **Deletion vs imputation?** Deletion biases data if not MCAR; imputation adds noise if wrong; understand why values are missing; use training stats only.

---

## Session 2.2 — Model Serving: REST, gRPC, and Streaming

Subtitle: "Prediction Modes • Latency & Throughput • Deployment Strategies". Prerequisites: Week 1 Sessions 1 & 2, REST APIs, basic systems thinking. Source chapter: Chip Huyen DMLS Ch. 7.
Framing: "how does a model actually receive requests, compute predictions, and return answers — reliably, at scale, with measurable latency guarantees? We move from theory to architecture."

Agenda:
1. The Reality of Model Deployment
2. Batch vs. Online Prediction
3. Latency, Throughput, and SLOs
4. Serving Patterns and Deployment Strategies

Closing slide: "**P99 latency in production: not your laptop benchmark.**"

### TL;DR
- **Four deployment myths** (Chip Huyen Ch. 7): you only have a couple of models; performance stays the same if you do nothing; you won't need to update much; scale is only for FAANG. All false.
- **Prediction modes**: batch (pre-computed, async) vs online (on-demand, sync); three practical modes — batch prediction; online prediction with batch features; streaming prediction with batch + streaming features. **Streaming features ≠ online features.**
- **Latency vs throughput**: batching raises throughput at the cost of latency (50 queries in 20 ms = 2,500 QPS vs 1 in 2 ms = 500 QPS). **Measure p50/p95/p99, never just the mean; set the SLA on p99.**
- **Serving patterns**: synchronous REST, asynchronous batch, streaming (Kafka/Flink), edge/on-device.
- **Deployment strategy = risk management**: shadow (prove correctness, zero user impact), canary (limit blast radius), blue-green (instant rollback), A/B (causal business impact). Typical pipeline: **shadow → canary → A/B → full rollout**. Tools: BentoML, NVIDIA Triton.

### Concepts (in lecture order)

#### §1 The Reality of Model Deployment

##### Myth 1 — "We only have a couple of models"
- The myth: "We deploy 1 or 2 models"; "Model management is not a big deal"; "One inference server is enough".
- Reality: **Uber: thousands** of models in production; **Google: thousands training concurrently**; **Netflix: ML for 20+ tasks** — content valuation & acquisition, churn prediction & retention, fraud detection, machine translation (subtitles), content tagging & search, personalised thumbnails, …and 14+ more.
- A ride-sharing app alone needs: ride demand, driver availability, ETA, dynamic pricing, fraud detection, churn. **20 countries × 10 models each = 200 models.**
- Post-handout: infrastructure must assume many models, independent update schedules, concurrent traffic — not a single notebook inference call.

##### Myth 2 — "Do nothing, performance stays the same"
- Software ages poorly; **ML models age worse**. A model performs best immediately after training.
- Data distribution shifts: user behaviour, world events, seasonality. **No retraining = silent degradation.**

##### Myth 3 — "We won't need to update much"
- Wrong question: "How often **should** I update?" Right question: "How often **can** I update?"
- Industry cadences: **Weibo**: some models updated every **10 minutes**; **Netflix**: **thousands of pushes per day**; **AWS**: a new deployment every **11.7 seconds**.

##### Myth 4 — "Scale is only for FAANG"
- Most ML engineers work at companies with 100+ employees; a 100-employee company likely has a meaningful number of users; the model runs **per request, not per company**.
- **10,000 users × 5 predictions/day = 50K daily inferences**; add peak traffic spikes and the problem becomes real fast.
- What scale concerns actually are: **Latency under load** (does p99 stay <100 ms at 10× traffic?); **Cost efficiency** (GPU utilisation at off-peak hours); **Reliability** (what happens when one replica crashes?); **Version management** (which model version is running where?).
- "An ML engineer should care about scale." — Chip Huyen, Ch. 7.

##### The degradation curve — models age poorly
- Why performance degrades:
  - **Data drift**: input distribution **P(X)** shifts
  - **Concept drift**: relationship **P(Y|X)** changes
  - **Label shift**: output distribution **P(Y)** changes
  - User behaviour changes seasonally; product features change, creating new patterns
- Domain-specific decay rates: **News ranking — hours**; **Fraud detection — days–weeks**; **Credit scoring — months**; **Medical imaging — may be stable for years**.
- Diagram: model quality vs time since training, curves for news (hours), fraud (weeks), credit (months) each dropping below an **SLA** line at different speeds ⇒ retraining cadence must match domain decay.

#### §2 Batch vs. Online Prediction

##### Two fundamental prediction modes
| Online prediction (on-demand) | Batch prediction (pre-computed) |
|---|---|
| Generated and returned as soon as request arrives | Generated periodically, stored, retrieved when needed |
| Also called **synchronous** or real-time prediction | Also called **asynchronous** prediction |
| User sends input → model computes → response returned | Predictions exist **before** the user asks |
| Requests via REST API or gRPC | Stored in data warehouse / cache |
| e.g., Google Translate returns French immediately | e.g., **Netflix generates recommendations every 4 hours** for all users, fetched at login |

Key distinction: "Online prediction waits for a request. Batch prediction runs ahead of time." Choice depends on latency requirement and data freshness needs.
Post-handout nuance: "**Online and batch are not the same as synchronous and asynchronous**… A model can serve online predictions using batch-computed features. Do not conflate the serving mode with the feature freshness."

##### Three prediction modes in practice
```
Mode 1: Batch prediction (batch features only)
  Data Warehouse -> Batch Prediction -> Cache/DB  <- User Request

Mode 2: Online prediction (batch features only)
  Live Request -> Online Model (features from Data Warehouse) -> Response

Mode 3: Streaming prediction (batch + streaming features)
  Live Request -> Online Model (features from Data Warehouse + Event Stream) -> Response
```

##### DoorDash: when you need both feature types
- Problem: predict delivery time (ETA) right now when user places an order; ETA depends on restaurant prep time + driver availability + traffic.
- **Batch features (historical)**, source: data warehouse, computed overnight — mean prep time of this restaurant; average driver availability in this area; historical traffic patterns for this route.
- **Streaming features (real-time)**, source: event stream — orders at this restaurant in the last 10 minutes; drivers available right now; unusual traffic at this moment.
- **Important clarification: Streaming features ≠ Online features.** Online features = any feature used in online prediction (including pre-computed batch features). Streaming features = computed exclusively from live event streams.

##### Batch vs. online — trade-offs
| Dimension | Batch prediction | Online prediction |
|---|---|---|
| Latency | Not user-facing; irrelevant | User is waiting; critical (**<500 ms typical**) |
| Throughput | High — millions at once | Limited by server capacity per request |
| Freshness | Stale by design — hours or days old | Fresh — computed at request time |
| Feature complexity | Can use slow, expensive features | Must use features retrievable in milliseconds |
| Infrastructure | Spark, Airflow, scheduled jobs | REST/gRPC servers, load balancers, caches |
| When to use | Weekly recs, monthly churn scores | Fraud check, search ranking, chatbots |

Break slide: "Your model can wait. You, however, need a break."

#### §3 Latency, Throughput, and SLOs

##### Latency vs. throughput — the core trade-off
- **Latency**: time from request sent to response received. **NOT training time.** Affected by model size, feature lookup time, preprocessing, network. "What users feel."
- **Throughput**: queries processed per unit time (**QPS**). "What determines your server cost."
- **Throughput ≠ 1/latency when batching.**
- **The batching paradox**: one query at a time ⇒ higher latency = lower throughput; with batching, you can increase throughput by accepting slightly higher latency.
  - 50 queries in a 20 ms batch: 50/0.020 = **2,500 QPS**
  - 1 query in 2 ms: 1/0.002 = **500 QPS** ⇒ 5× throughput (pre-handout: "accepting slightly higher latency can give you 5× the throughput").

##### Why percentiles, not means
10 latencies (ms): `[100, 102, 100, 100, 99, 104, 110, 90, 3000, 95]`
- Mean ≈ **390 ms** (misleading — dominated by the 3-s outlier)
- p50 ≈ 100 ms; p95 ≈ 110 ms; p99 ≈ 3000 ms
- 9 of 10 users had a fine experience; 1 in 10 waited 3 seconds.
- **Why p99 matters most**: p99 = the 1% of users with the worst experience; **at 1M QPS that's 10,000 users/second** experiencing a broken product. Large companies often target **p99.9**. Mean-based SLAs can hide catastrophic tail latency.
- **Real-world impact**: **Akamai (2017): 100 ms delay → 7% drop in conversion.** **Booking.com (2019): +0.5% latency → −0.5% conversion** ("0.5% latency cost").

##### Latency distribution — setting SLOs
- Diagram: right-skewed latency histogram, markers in order p50 < mean < p95 < p99 (mean pulled right by the tail).
- **How to set SLOs**: (1) understand user expectations, (2) measure current distribution, (3) set realistic targets at multiple percentiles.
- **Common ML SLOs**:
  - p50 latency: typical user experience target
  - p95 latency: most users are served this well
  - **p99 latency: worst 1% — often the SLA threshold**
  - Throughput: minimum QPS under peak load
  - Availability: uptime % (**99.9% = 8.7 hrs downtime/year**)
  - Error rate: **<0.1%** failures
- Recap slide: "Setting SLOs before choosing model size."

##### Quiz 1 (with answers)
- **Q1 Fraud detection serving mode**: flag in **<50 ms**, every card transaction, **10,000 TPS peak**. → **Online (synchronous) prediction**; batch impossible (decision needed before transaction completes). Challenges: (1) feature lookup latency — Redis online store keeps lookups **<5 ms**; (2) at 10K QPS horizontally scale the serving fleet with load balancing; (3) p99 — one slow GPU call breaches the SLA ⇒ **request timeouts and fallback rules**.
- **Q2 Diagnosing latency**: average 200 ms "fine", paged at 2am, **p99 = 4,000 ms**. → 99% fine, 1% take 4 s: a **tail latency problem** (long-tail event, not general slowdown). Investigate: (1) p99 correlated with a specific feature lookup (slow DB query on rare users)? (2) **cold-start** (first request loads the model)? (3) downstream service timing out on 1% of requests? Check **per-component latency breakdowns in your tracing tool**.

#### §4 Serving Patterns and Deployment Strategies

##### Four serving patterns
| Pattern | Use when | Examples |
|---|---|---|
| **Synchronous REST** | User is waiting; latency < 1 s; single item; simple ops | Fraud check at checkout, spam filter, search ranking |
| **Asynchronous batch** | High volume; no real-time need; cost-sensitive; overnight or hourly | Weekly recommendations, monthly churn scores, demand forecasts |
| **Streaming (Kafka/Flink)** | Continuous events; sub-second freshness; stateful aggregation | Real-time fraud, dynamic pricing, live anomaly detection |
| **Edge / on-device** | Privacy constraints; offline use; ultra-low latency **<10 ms** | Face unlock, keyword spotting, autonomous vehicle perception |

##### Shadow, canary, blue-green
| Shadow | Canary | Blue-green |
|---|---|---|
| New model runs in parallel | **5–10%** of traffic → new model | Two identical environments |
| Does **not** serve real traffic | Monitor real metrics on real users | Switch **100%** of traffic at once |
| Predictions logged and compared offline | Gradually increase if stable | Old environment stays warm |
| Safe — **zero user impact** | Roll back fast if metrics drop | **Instant rollback: flip the switch** |
| Cost: **doubles compute** | | |
| Best for: high-risk models, first validation step | Best for: incremental model updates, measured rollout | Best for: major version upgrades, hard cutover requirements |

Post-handout: "**Deployment strategy is risk management.** Shadow proves correctness before any user is affected. Canary limits blast radius. Blue-green enables instant rollback. A/B provides causal evidence of business impact. Choose based on risk tolerance and what you need to prove."

##### A/B testing as a deployment strategy
- Different models serve different user segments (diagram: all user traffic → 50% Model A (control) / 50% Model B (variant) → metrics comparison).
- Compare **business metrics**: click rate, conversion, retention.
- **Randomised assignment ⇒ causal claims** about model quality. Not just for UI changes — essential for model comparison.
- **Statistical significance required** before declaring a winner.
- **A/B vs canary**: canary = reduce deployment risk, same model goal; A/B = measure business impact of two model strategies. Often run together: **canary → A/B → full rollout**.

##### BentoML and NVIDIA Triton Inference Server
- **BentoML**: wraps Python models into REST/gRPC services; works with sklearn, PyTorch, TensorFlow, XGBoost; built-in model versioning, **adaptive batching**, Docker packaging; deploy locally, cloud, or Kubernetes; "great for teams that want simplicity without sacrificing power". CLI: `bentoml serve model:svc --port 3000`.
- **NVIDIA Triton**: optimised inference for GPU and CPU; supports TensorFlow, PyTorch, ONNX, TensorRT; **dynamic batching** (groups concurrent requests automatically); **concurrent model execution** (multiple models in parallel); **model ensembles** (chain models into pipelines). Used at NVIDIA, Microsoft, LinkedIn, major cloud providers.
- Other tools: **TorchServe** (PyTorch-native), **Ray Serve** (Python-native distributed), **TF Serving** (TensorFlow-native), **Seldon Core** (Kubernetes-native).

##### Quiz 2 (with answers)
- **Q1 Choose deployment strategy**: new recommendation model, old one in prod 6 months, validate CTR impact before full rollout, **no downtime**. → **Shadow → Canary → A/B → full rollout.** Shadow: parallel, log, compare offline, zero user risk. Canary (5–10%): catch infra issues (latency, error rates). A/B: 50/50 split to measure CTR; declare winner at **p < 0.05**. Old model keeps serving throughout.
- **Q2 Batch vs streaming features** (surge pricing every minute): "average ride demand in area over past 30 days" (nightly) = **batch** — nightly Spark job → offline store → sync to Redis. "Ride requests in the area in past 5 minutes" = **streaming** — Kafka + Flink rolling counter per area → Redis in real time. Architecture: batch = **data warehouse + Spark + scheduler**; streaming = **Kafka + Flink + Redis**.

##### Recap slide
§1: 4 myths; models degrade without retraining; Weibo/Netflix/AWS cadences; scale matters even at 100 employees. §2: three prediction modes; DoorDash batch + streaming; streaming ≠ online features; trade-off table freshness vs cost. §3: latency ≠ throughput, batching paradox; measure p50/p95/p99 not mean; Akamai & Booking.com; set SLOs before choosing model size. §4: 4 serving patterns; shadow/canary/blue-green; A/B for model comparison; BentoML and Triton.

### Demo notebook — `w2s2_demo_model_serving.ipynb` (instructor-run)
Four parts: (1) sklearn model as FastAPI endpoint, (2) latency percentile problem, (3) batch vs online prediction, (4) BentoML.

**Part 1 — FastAPI endpoint.** Trains an Iris `Pipeline([StandardScaler(), LogisticRegression(max_iter=500)])`, pickles to `/tmp/iris_pipeline.pkl`. Serving code (`serve.py`, run with `uvicorn serve:app --host 0.0.0.0 --port 8000`; shown read-only):
```python
app = FastAPI()
with open('/tmp/iris_pipeline.pkl', 'rb') as f:
    MODEL = pickle.load(f)            # load once at startup
class PredictRequest(BaseModel): sepal_length: float; ...   # Pydantic typed I/O
@app.post('/predict', response_model=PredictResponse)
def predict(request): proba = MODEL.predict_proba(features)[0]  # -> predicted_class, confidence, probabilities
@app.get('/health')
def health(): return {'status': 'ok', 'model_loaded': MODEL is not None}
```
**Key design decisions**:
1. **Model loaded at startup** — loading inside `predict()` re-reads disk per request; "at 1000 req/s that's catastrophic".
2. **Typed request/response** — Pydantic gives free input validation and API docs.
3. **Health endpoint** — load balancers use `/health` to route traffic; "without it, a crashed pod still receives requests".
4. **Probabilities returned** — downstream systems often need the full distribution.

**Part 2 — Latency percentile problem** ("why 'average latency = 12ms' tells you almost nothing"). Simulates 1,000 requests: 97% lognormal(mean=2.5, σ=0.4) ms + **3% uniform 200–1000 ms outliers (GC pauses, cold cache, model reloading)**. Prints mean, p50, p75, p90, p95 ("1 in 20"), p99 ("1 in 100"), max; compares with/without outliers: the **mean barely moves while p99 jumps by hundreds of ms** — "the mean hides this completely". At 100 req/s, 1% slow = 1 bad experience per second. Histogram plots (full + zoom < 100 ms) with mean/p50/p95/p99 lines, titled "Serving Latency: Mean Hides the Tail". Recommendation: **monitor p50, p95, p99 — not mean.**

**Part 3 — Batch vs online prediction** (recommender: 10,000 users, 500 items, 20 features, top-10; simplified **two-tower dot product** `item_feats @ user_feat` + `np.argpartition`). Batch: precompute top-10 for all users into a dict ("in prod: Redis or DynamoDB"), serving = dict lookup. Online: compute on request. Comparison table printed:
| | Batch | Online |
|---|---|---|
| Serving latency | dict lookup (tiny) | compute on demand |
| Staleness | up to 24h | 0 ms (real-time) |
| New user handling | **fails** | works immediately |
| Feature freshness | stale | fresh |
| Compute cost | amortized | per-request |
When batch: recommendations, reports, email campaigns, low-traffic endpoints. When online: search, fraud detection, dynamic pricing, real-time feeds.

**Part 4 — BentoML walkthrough** (conceptual). FastAPI is great for small services; production ML serving wants model versioning/registry, **adaptive batching**, **Runner APIs** (separate CPU pre/post-processing from GPU inference), Docker/K8s packaging.
Steps: (1) `bentoml.sklearn.save_model('iris_classifier', pipeline)` → versioned model store; (2) define Service:
```python
iris_runner = bentoml.sklearn.get('iris_classifier:latest').to_runner()
svc = bentoml.Service('iris_service', runners=[iris_runner])
@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(features):
    proba = await iris_runner.predict_proba.async_run(features)   # adaptive batching handled here
```
(3) `bentoml build` → Bento (model + service + deps); (4) `bentoml containerize iris_service:latest` → Docker image. Concepts: **Runner** (async inference wrapper enabling adaptive batching), **Service** (API endpoints + runners), **Bento** (deployable package), **Store** (versioned model registry).
**Why adaptive batching matters**: a GPU processes 64 examples in almost the same time as 1; one-at-a-time uses 1/64th of capacity. Runner waits a configurable few ms (e.g., **5 ms**), batches, runs once, returns to each caller ⇒ at 1000 req/s can multiply throughput **10–50×** with negligible latency increase.
**Discussion**: describe a system using both batch precompute and online compute — when to serve from cache vs fall back to online? (note: e.g., serve precomputed recs for known users; fall back to online for new users / stale entries — matches the "new user handling: fails" row.)

### Pre/Post handout extras
**Pre-handout session plan**: opening "how would you serve a model to 1 million users?"; myths discussion ("which ones have you believed?"); three prediction-mode architecture diagrams; DoorDash example; latency quiz (p99); shadow vs canary vs blue-green comparison; hands-on BentoML look; two quizzes + class discussion. Prep: think how you'd serve a model to 1M users — what breaks first? Recall REST end-to-end; optionally look up gRPC vs REST; skim DMLS Ch. 7 (prediction modes, myths). No installs needed.

**Post-handout key takeaways** (4): production is many models, not one; online/batch ≠ synchronous/asynchronous (don't conflate serving mode with feature freshness); measure latency in percentiles (at 1M QPS p99 of 4,000 ms = 10,000 users/sec broken; set SLA on p99, not the average); deployment strategy is risk management.

**Test Your Understanding (with handout answers)**:
- Q1: model deployed 6 months ago, no monitoring/retraining, business metrics quietly declining → contradicts **Myth 2** (do nothing, performance stays the same) and **Myth 3** (won't need to update much). Should have had: (1) **monitoring pipeline** tracking input feature distributions and business metrics at serving time with automated drift alerts; (2) **scheduled (or drift-triggered) retraining pipeline**.
- Q2: 100 QPS currently; dynamic batching of 20 requests in 18 ms vs old 1 request in 2 ms. (a) throughput = 20/0.018 ≈ **1,111 QPS**; (b) latency per request = **18 ms** (all wait for the batch); (c) depends on SLA — fine if 18 ms tolerated (e.g., background API; ~11× throughput vs sequential), bad if users expect <5 ms. "**The batching paradox only applies when the latency increase is within your budget.**"
- Q3: fraud model — (a) **shadow**: run in parallel, log both predictions per transaction, compare offline, no user gets new decision; (b) **canary**: route 5% of live transactions to new model, monitor false positive/negative rates, increase if stable; (c) **blue-green**: complete new environment, both running, switch 100% at once, keep old warm for instant rollback.
- Q4: **online feature** = any feature used at online prediction time (can be pre-computed batch value), e.g., avg ride demand over past 30 days (nightly) — an online feature that is also a batch feature; **streaming feature** = computed from live event stream, e.g., ride requests in this area in last 5 minutes (Kafka).

**Exercises**:
- Ex 1: Wrap a model with BentoML; measure latency of 100 sequential requests with `time`; compute p50 and p99; paragraph on how to improve p99.
- Ex 2: **Latency budget analysis** for an app you use (Google Search, Spotify, Instagram): (a) acceptable end-to-end time, (b) budget used by network/pre/post-processing, (c) what's left for the model; sketch breakdown.
- Ex 3: **Prediction mode decision** — (1) batch, (2) online w/ batch features, (3) online w/ streaming features for: (a) monthly churn scores for SaaS, (b) real-time spam filter for email server, (c) dynamic ad bidding using user's last 3 clicks, (d) personalised homepage recs at login. (note: likely intended — (a) batch; (b) online w/ batch features (content of the email is the request input); (c) online w/ streaming features; (d) batch precompute or online w/ batch features.)
- **Challenge**: recommendation serving architecture for a streaming platform, **5M DAU**, SLA **recs load within 100 ms of page open**: (1) prediction mode justified by 100 ms SLA; (2) ≥2 batch and ≥2 streaming features; (3) deployment strategy for next quarter's model; (4) three SLO metrics + percentiles; (5) **fallback** when serving infra goes down.

**Further reading**: DMLS Ch. 7 (myths, batch vs online, three modes — "essential reading") and Ch. 1 (latency vs throughput, percentiles); BentoML docs (getting started, serving, batching, Docker/K8s); Triton docs (dynamic batching, model ensembles, performance analyser). Online: Chip Huyen's blog, Martin Fowler — Blue-Green Deployment (canonical reference), Akamai 2017 study (100 ms → 7%), Booking.com 2019 experiment, Google SRE Book Ch. 4 (SLOs and SLAs), The Morning Paper (blog.acolyer.org).

### Likely doubts & exam angles
1. **Is "online prediction" the same as "uses streaming features"?** No. Online = computed at request time; it can use only batch features (Mode 2). Streaming prediction (Mode 3) adds features from live event streams.
2. **Why is throughput ≠ 1/latency?** With batching, many requests share one forward pass: 50 in 20 ms = 2,500 QPS vs 1 in 2 ms = 500 QPS, even though per-request latency is 10× higher.
3. **Why not report mean latency?** Skewed distributions: one 3,000 ms outlier makes mean ≈ 390 ms while p50 ≈ 100 ms. Tail users are invisible; set SLA on p99 (big cos: p99.9).
4. **What does 99.9% availability mean?** ~8.7 hours downtime/year.
5. **Shadow vs canary?** Shadow serves no real users (zero impact, doubles compute, validates correctness). Canary serves 5–10% real users (real metrics, limits blast radius).
6. **Canary vs A/B?** Canary = risk reduction for rollout of the same goal; A/B = randomised, statistically significant measurement of business impact (CTR/conversion), declare at p < 0.05.
7. **When blue-green?** Major version upgrades / hard cutovers needing instant rollback; costs two full environments.
8. **Which serving pattern for <10 ms / privacy / offline?** Edge / on-device.
9. **Why does a model degrade?** Data drift P(X), concept drift P(Y|X), label shift P(Y); decay rate depends on domain (news hours → medical years).
10. **First things to check for high p99?** Slow feature lookups for rare users, cold start, downstream timeouts — use per-component tracing.
11. **Why load the model at startup and add /health?** Avoid per-request disk reads; let load balancers stop routing to crashed pods.

---

## Cross-links
- **Week 1 → Week 2**: W1S1 introduced the six components of a production ML system and failure modes; W2S1 deep-dives the data/feature component; W2S2 the serving/deployment component (pre-handouts state this).
- **Within Week 2**: the S1 offline/online feature store is exactly what S2's prediction Modes 2 and 3 read from (Redis online store <5 ms lookups in the fraud quiz); both sessions reuse the same batch-vs-streaming feature quiz pattern (30-day window ⇒ Spark/warehouse; 5-min window ⇒ Kafka+Flink+Redis).
- **Forward (monitoring / retraining)**: S2's degradation curve (data drift, concept drift, label shift) and post-handout answer (monitoring pipeline + drift-triggered retraining) set up later monitoring/continual-learning topics; S1's feature store "monitoring" (offline vs online distribution drift) feeds the same theme.
- **Forward (LLMOps serving)**: dynamic/adaptive batching (Triton, BentoML), p99 SLOs, and shadow/canary/A/B rollouts recur for LLM inference serving. (note: inferred from course title; check later weeks' notes.)

## Glossary
- **Row-major** — storage layout with rows contiguous on disk (CSV, JSON); fast writes, fast whole-record reads.
- **Column-major** — columns contiguous on disk (Parquet, ORC); fast single-feature reads, strong compression.
- **Parquet** — column-major binary format; ~6× smaller and ~2× faster than CSV on S3.
- **Column pruning / predicate pushdown** — reading only needed columns / filtering at the storage layer (Parquet + Spark).
- **Structured / unstructured data** — schema'd tabular data (10–20%) vs text/images/audio etc. (80–90% of industry data).
- **First/second/third-party data** — your users / another company's customers / aggregated public sources.
- **IDFA** — Apple's advertising identifier; made opt-in in 2021, hitting the $80B ad industry.
- **Normalization (DB)** — 1NF/2NF/3NF schema design to remove redundancy (Harry Potter publisher example).
- **ACID** — transactional guarantees offered by relational DBs.
- **NoSQL types** — document, key-value, graph, column-family.
- **OLTP** — transactional, row-oriented, writes/point lookups (Postgres, MySQL, Aurora).
- **OLAP** — analytical, column-oriented, aggregate scans (Redshift, BigQuery, Snowflake, ClickHouse).
- **ETL** — extract-transform-load; nightly movement OLTP → OLAP.
- **REST** — JSON over HTTP, synchronous request-response.
- **gRPC** — RPC with Protocol Buffers (binary), faster than REST.
- **Kafka** — pub/sub data broker; producers write to topics, consumers read; decouples services.
- **Batch processing** — periodic processing of accumulated data (Spark, MapReduce); high throughput, high latency.
- **Stream processing** — per-event processing (Flink, Spark Streaming, Kafka Streams); low latency, stateful aggregation is hard.
- **MCAR** — missing completely at random; deletion is safe only then.
- **Imputation** — filling missing values (mean/median/mode/model), using training stats only.
- **Min-max normalization** — x' = (x−min)/(max−min); bounded, no outliers.
- **Standardization** — x' = (x−μ)/σ; Gaussian-ish, with outliers.
- **Feature crossing** — combining features to capture interactions (marital status × #children).
- **Data leakage** — information unavailable at prediction time leaks into training (scaling before split, future info, label proxies, group leakage).
- **Temporal leakage** — random split on time-ordered data letting the model see the future.
- **Point-in-time correctness / join** — for a training row at time t, only use feature values computed from data < t.
- **Training-serving skew** — feature values differ between training and serving due to divergent computation.
- **Feature store** — system defining features once and serving them consistently to training (offline) and serving (online); "a contract between data engineering and model serving".
- **Offline store** — historical features for training (S3 + Parquet/Delta Lake; Spark/dbt).
- **Online store** — latest features for inference (Redis/DynamoDB; sub-ms).
- **Staleness budget** — how old an online feature value is allowed to be.
- **Materialization** — computing and writing feature values into the stores.
- **Feature view** — Feast's unit of feature definition.
- **Backfilling** — computing features for historical dates on demand.
- **Online prediction** — synchronous, on-demand prediction at request time.
- **Batch prediction** — asynchronous, pre-computed predictions stored and fetched later.
- **Streaming prediction** — online prediction using batch + streaming features.
- **Online feature** — any feature used during online prediction (may be batch-computed).
- **Streaming feature** — feature computed exclusively from live event streams.
- **Data drift** — P(X) shifts. **Concept drift** — P(Y|X) changes. **Label shift** — P(Y) changes.
- **Latency** — request-to-response time (what users feel). **Throughput** — QPS (what drives cost).
- **Batching paradox** — accepting higher per-request latency to gain throughput.
- **p50/p95/p99/p99.9** — latency percentiles; p99 = worst 1%, usually the SLA threshold.
- **SLO / SLA** — internal target / contractual commitment on latency, throughput, availability, error rate.
- **Tail latency** — latency of the slowest requests (p99+); caused by cold starts, GC, slow lookups, downstream timeouts.
- **Cold start** — slow first request(s) while the model/cache loads.
- **Shadow deployment** — new model runs in parallel without serving users; predictions logged and compared.
- **Canary deployment** — small slice (5–10%) of traffic to new model, ramp up if stable.
- **Blue-green deployment** — two identical environments; flip 100% traffic, keep old warm for instant rollback.
- **A/B test** — randomised split (e.g., 50/50) comparing business metrics with statistical significance.
- **Edge / on-device serving** — inference on device for privacy, offline use, <10 ms latency.
- **BentoML** — framework wrapping Python models into REST/gRPC services; Runner, Service, Bento, Store; adaptive batching.
- **NVIDIA Triton** — GPU/CPU inference server with dynamic batching, concurrent model execution, ensembles.
- **Adaptive / dynamic batching** — server waits a few ms to group concurrent requests into one forward pass.
- **Health endpoint** — `/health` route load balancers probe to decide routing.
- **Two-tower model** — user and item embeddings scored by dot product (used in the S2 demo recommender).
