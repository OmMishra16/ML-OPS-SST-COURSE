# Week 8 — Agentic Systems Design + Bonus Case Studies (Ride-Hailing, Coding Agent) + Course Closing
Sources (originals in `/Users/ommishra/Desktop/ML/class_content/week8/`): w8s1_student_worksheet.pdf, w8s1_solution_design.pdf, w8_Ride_hailing_case_study_bonus.pdf,
w8_Coding_agent_case_study_bonus.pdf, course_closing_slides.pdf

(note: Week 8 has no lecture slide deck or demo notebook. Session 1 is a system-design workshop (AI customer-support
agent) run on the 10-step ML System Design Rubric (full rubric table is in week7.md). The two bonus PDFs are labelled
"Week 8, Session 3/4, Bonus Material — not part of the course", trimmed excerpts that each work the rubric end-to-end.)

---

## Session 8.1 — System Design Workshop: Multi-model Agentic Pipeline (AI Customer Support)

### TL;DR
- Design a production AI customer-support agent for e-commerce: **2M customers, 10K concurrent conversations at peak,
  p99 < 5 s/turn, cost ceiling $0.05 per resolved conversation, audit trail for every action, human escalation within
  60 s, graceful handling of prompt injection.**
- **Key insight: "Agents fail in production in ways models don't."** Four failure modes: (1) unbounded loops, (2) cost
  blow-up, (3) context overflow, (4) tool failure cascade. **"Design for bounded execution, not optimal execution."**
- Architecture: language detect/translate → **DistilBERT intent router (< 20 ms)** → (direct tool | KB+LLM | full agent
  loop | escalate) → **GPT-4o-mini with function calling** → guardrails (PII/policy/injection, human approval for refunds
  > $50, cost ceiling $0.10/session, loop detector) → response or escalation; **immutable, PII-scrubbed audit log**.
- **Hard limits in the agent runner:** max **15 turns**, max **$0.10/session**; same tool + same args twice → break & escalate.
- Severity bias in A/B: AI resolves easy cases, humans get hard ones → inflated AI metrics; **track resolution per intent**.
- Know **when NOT to use an agent**: rule-based suffices, < 100 ms latency, high-stakes irreversible actions without override.

### Problem (from worksheet)
> Design a production AI customer support agent for an e-commerce platform.
> **Scale:** 2 million customers, 10,000 concurrent conversations at peak.
> **Requirements:** p99 latency < 5s per turn · handles order status, returns, product questions, complaint escalation ·
> cost ceiling $0.05 per resolved conversation · audit trail for every action · human escalation within 60s · graceful
> handling of prompt injection.

**W8 rubric variant (agent-specific wording):**
1. Clarify — incl. "What is the escalation path when the system fails?"
2. Context — "What is the cost of a **wrong action**?"
3. Problem Statement — input, output, and "what decision does the agent need to make at each step?"
4. Requirements — Functional (intent classification, tool use, escalation); Non-functional (latency SLA, scale, cost
   ceiling, auditability, availability).
5. Estimations — concurrent sessions, tokens/day, LLM cost/hour, tool call volume — "These numbers expose whether the
   design is financially feasible."
6. Architecture — (a) Data Collection: conversation history, tool results; (b) Data Engineering: state management,
   context window; (c) Feature Engineering: customer context, KB; (d) Preprocessing: **intent routing**; (e) Models:
   classifier + LLM orchestrator + tools; (f) Post-processing: **safety, audit, escalation**.
7. Evaluation — Offline: intent accuracy, tool selection accuracy, resolution rate on golden test set. Online: resolution
   rate, CSAT, cost per conversation, escalation rate.
8. A/B — **severity bias** (AI resolves easy cases, humans get hard ones — inflates AI metrics). Track both.
9. Deployment — stateless pods + Redis session state; monitor turns-per-conversation, cost/conversation, tool API error
   rates, escalation rate by intent, CSAT rolling average.
10. Risk — autonomous vs human approval for high-stakes actions; **loop detection (max turns + repeated action detection
    + cost ceiling)**; prompt-injection defences; **when NOT to use agents**.
- "For agentic systems, **Step 10 is especially important** — agents fail in production in ways models don't."

Worksheet tables: Step 4 rows — Latency SLA per turn, Concurrent sessions, Cost ceiling, Autonomous action scope, Audit
trail. Step 5 rows — LLM tokens/hour, LLM cost/hour, Tool API calls/min, Session state size in Redis. Step 6 —
"**CRITICAL: where do you enforce max-turn limit and cost ceiling?**"; boxes (a–b) State Management & Data Engineering,
(c–d) Customer Context & Intent Routing, (e–f) Models, Tools & Safety + full agent flow ("classify → route → tool call →
LLM → response. Mark where loop detection and cost ceiling fire."). Step 9 asks for **8 metrics**. Step 10 rows:
Autonomous vs human approval for high-stakes actions; Single LLM vs routing to specialised models; Full history vs sliding
window context; Loop prevention strategy; When NOT to use an agent here. Closing prompt: **"An angry customer sends 50
messages in 10 minutes. What happens?"** — trace what fires, what limits apply, how the customer ends up with a human.

### Concepts (reference solution, in rubric order)

#### Step 1 — Clarifying questions
| Question | Answer |
|---|---|
| Can the agent take autonomous actions (refunds) or only recommend? | **Autonomous for low-risk; human approval for refunds > $50** |
| Escalation SLA to a human? | Within **60 seconds** of trigger |
| Languages? | English + top 5 by user volume |
| What defines "resolved"? | **Customer did not re-contact within 24h** |
| Audit requirement? | Yes — every action logged for compliance |

#### Steps 2 & 3 — Context and Problem Statement
- **Context:** support is a cost centre being partially automated. Wrong actions (unwarranted refunds, sharing wrong
  order details, not escalating an angry customer) cause direct financial loss and trust damage. **"The agent must be
  conservative — better to escalate than to act incorrectly."**
- **Problem statement:** Given a customer message + conversation history, classify intent, retrieve context (order data,
  product info, policies), and either resolve via API action or generate a helpful response. If resolution fails or
  confidence is low → escalate.
  - Input: message + session context + customer ID.
  - Output: response text + optional action taken + **escalation flag** + **audit log entry**.

#### Step 4 — Requirements
| Requirement | Implication |
|---|---|
| **Functional** | |
| Intent classification | order status / returns / product questions / complaints / other |
| Tool use | order lookup, refund API, product KB search, escalation trigger |
| Multi-turn conversation | session state must persist across turns (**Redis**) |
| Audit log of every action | every tool call + result + LLM response logged **immutably** |
| Language detection + translation | pre-processing step before intent classification |
| **Non-functional** | |
| p99 < 5 s per turn | LLM + tool calls must fit within budget |
| 10,000 concurrent conversations | **stateless pods; state in Redis** |
| Cost ceiling $0.05/resolved conversation | per-conversation cost monitor with **hard cutoff** |
| Human escalation within 60 s | escalation queue with SLA monitoring |

#### Step 5 — Estimations
| What | Estimate |
|---|---|
| Sessions | 10K concurrent × avg 5 turns = 50K turns per session-set |
| LLM tokens | 10K concurrent × 1K tokens/turn × 5 turns = **50M tokens/hour** |
| LLM cost | at $0.15/1M tokens = **$7.50/hour** (LLM alone) |
| Tool calls | avg 2 tool calls/conv × **3,333 new conv/min** = **6,666 tool API calls/min** |
| Session state | 10K sessions × 10 KB = **100 MB in Redis** (easily fits) |
| Cost per conversation | $7.50/hr ÷ (3,333 conv/min × 60) ≈ **$0.037/conv — within $0.05 ceiling** (as stated in source; note: the literal division gives ≈ $0.0000375; the source's conclusion "within ceiling" holds either way, but if asked, note the arithmetic) |

#### Step 6 — System Architecture
**Key insight:** Agents fail in production in ways models don't. Four failure modes:
1. **Unbounded loops** — agent calls tools forever;
2. **Cost blow-up** — one bad query runs up $50 in LLM calls;
3. **Context overflow** — conversation grows until it hits the context window;
4. **Tool failure cascade** — order API timeout stalls the entire agent.
→ **"Design for bounded execution, not optimal execution."**

**6a Data Collection & State Management**
- Sources: customer messages (chat API), order data (OMS), product data (catalogue API), policies (internal KB),
  conversation history (**Redis, TTL 24h**).
- **Context window management:** sliding window keeps last N turns; older turns summarised into a compressed memory
  string. **Hard cap: session > 15 turns → force escalation.**
- **Customer context:** order history, account tier, past issues fetched at session start and injected into system prompt;
  does not change during session.

**6b & 6c Data Engineering & Feature Engineering**
- **Tool result caching:** order status **30 s**; product info **5 min**; policy answers **1 h**.
- **Product KB:** descriptions + FAQs embedded (text-embedding-3-small), indexed in **FAISS**; queried for product intents.
- **PII scrubbing:** all conversation logs scrubbed (card numbers, addresses) before writing to audit store.

**6d & 6e Preprocessing / Intent Routing & Models**
- **Intent classifier:** fine-tuned DistilBERT (**< 20 ms**), 5 intents with routes:
  | Intent | Route |
  |---|---|
  | order_status | **direct tool call (no LLM needed)** |
  | product_question | KB retrieval + LLM |
  | complaint | full agent loop |
  | other | **immediate escalation** |
  | returns | (note: route not spelled out in source; implied tool path via `initiate_return` + LLM) |
- **Main LLM:** GPT-4o-mini with function calling. Tools: `get_order_status(order_id)`, `initiate_return(order_id,
  reason)`, `issue_refund(order_id, amount)`, `search_product_kb(query)`, `escalate_to_human(reason)`.
- **Translation:** Helsinki-NLP/opus-mt for non-English input (pre) and output (post); adds **~100 ms**.
- **Loop detection:** same tool with same arguments twice in one session → break loop and escalate.

**6f Post-processing & Guardrails**
- **Action confirmation:** refunds **> $50 require human approval** — agent says "I can initiate this refund but it
  requires manager approval — I've triggered that process."
- **Safety checks:** PII in response (block); policy violation (block + escalate); prompt-injection detection ("ignore
  previous instructions" patterns).
- **Audit log:** every tool call + arguments + result + LLM response + latency written immutably to an **append-only store**.
- **Cost ceiling:** accumulated LLM token cost for session **> $0.10 → force graceful escalation**.
  (note: the *per-session hard cutoff* is $0.10 while the *business target* is $0.05 per resolved conversation and the
  *alert* threshold is $0.08 — three different numbers serving different purposes.)

**End-to-end diagram (reconstructed lanes):**
```
6a SOURCES        6d PREPROCESS            6a/6c STATE                 6e MODELS                 TOOLS                  6f GUARDRAILS                 OUTPUTS
Customer msg  ->  Language detect      ->  Redis session (TTL 24h) ->  Intent router        ->  get_order_status   ->  Safety: PII/policy/injection ->  Response to customer
(chat API)        + translate              Sliding window + summ.      Main LLM GPT-4o-mini     Tool layer             Human approval gate (>$50)  ->  Escalate to human (<60s SLA)
Order data (OMS)  Intent classifier        (cap 15 turns)              + function calling       issue_refund /         Cost ceiling $0.10/session  ->  Audit log (append-only,
Product cat. API  (DistilBERT, <20ms)      FAISS product KB            Loop detector            initiate_return                                         PII-scrubbed)
Policy KB                                  Tool result cache           (repeated call => break)
                                           (30s / 5min / 1h)
                  [session state updated after each turn; loop-detector feedback into LLM]
```
"Reading the diagram: solid arrows = primary request flow; dashed = read-only context lookups and feedback. **Every
guardrail node has two exits: forward to the customer or sideways to escalation — this is what bounded execution looks
like structurally.**"

#### Step 7 — Evaluation
| Type | Metric | Why |
|---|---|---|
| Offline | Intent accuracy | Does the classifier route correctly? |
| Offline | Tool selection accuracy | Does the agent call the right tool? |
| Offline | Resolution rate on golden set | End-to-end test on labelled conversation set |
| Online | **Resolution rate (north star)** | % conversations resolved without human |
| Online | CSAT | post-conversation rating (1–5 stars) |
| Online | Cost per resolved conversation | must stay < $0.05 ceiling |

**Guardrails:** **Escalation rate** — **< 5% = overconfident; > 40% = too cautious.** **Avg turns per conversation > 8 =
stuck in loop.** **Cost per conversation alert > $0.08.**

#### Step 8 — A/B Testing
- **Unit: customer session** (not message — session context matters).
- Treatment vs control: new intent classifier / new LLM vs current production.
- **Duration: 7 days** — support is high volume; effects detectable faster than most ML systems.
- **Severity bias:** the most complex conversations get escalated regardless → AI-resolved pool looks easier than it is.
  **Track resolution rate separately per intent type.**
- **Decision rule:** promote if resolution rate ↑ **≥ 1%** AND CSAT does not drop AND cost/conversation stays **< $0.05**.

#### Step 9 — Deployment and Monitoring
**Deployment:** stateless agent pods behind load balancer; session state in Redis (not pod memory); **circuit breaker on
all external tool APIs**; **canary rollout for new LLM models (10% traffic first)**; **hard limits in agent runner: max 15
turns, max $0.10/session**.

**8 metrics to monitor:** (1) turns-per-conversation distribution; (2) tool API error rates by tool; (3) escalation rate
by intent type; (4) LLM cost per hour; (5) CSAT by conversation type; (6) PII leakage alerts; (7) prompt injection
attempts; (8) refund approval queue length.

#### Step 10 — Risk Management and Trade-offs
| Decision | Choice | Alternative | Why |
|---|---|---|---|
| Autonomous actions | **Tiered** (auto low-risk, human high-risk) | Fully autonomous | Fully autonomous risks unwarranted refunds and policy violations; tiered balances speed vs safety. |
| Single LLM vs multi-model | **Routing to specialised models** (classifier + main LLM) | One large LLM for everything | Intent classifier at 20 ms is **250× cheaper** than GPT-4o for every routing decision; **saves $5+/hr** at scale. |
| Context management | **Sliding window + compression** | Full history | Full history hits context limit after **~20 turns**; compression is lossy but necessary. |
| Loop prevention | **Max turns + repeated-action detection + cost ceiling** | None (rely on LLM to self-terminate) | **LLMs do not reliably self-terminate loops. Hard limits are non-negotiable.** |
| When NOT to use agents | High-stakes irreversible actions without override; latency < 100 ms; rule-based sufficient | Agents everywhere | Agents add cost and latency. "If order status = delivered, reply with tracking URL" does not need an agent. |

**Failure modes and recovery:**
| What breaks | Detect | Recovery |
|---|---|---|
| Infinite loop | turns > 8; same tool called twice | Hard cap 15 turns; repeated-action detector fires; force graceful escalation |
| Context window overflow | LLM returns `context_length_exceeded` | Sliding-window compressor; **summarise oldest 50% of turns** |
| Tool API timeout | latency spike; circuit breaker fires | Return cached result if available; inform customer of delay; escalate if tool needed for resolution |
| Prompt injection attack | known patterns in input; safety filter fires | Block response; log event; escalate to human; **flag customer account for review** |
| Cost blow-up | per-session cost tracker > $0.10 | Immediate forced escalation; alert on-call; investigate which intent type caused it |

#### Worksheet closer — "Angry customer sends 50 messages in 10 minutes"
(note: no model answer given; trace assembled from the solution.) Intent classifier tags `complaint` → full agent loop.
Turn count climbs: avg-turns > 8 alerting; **hard cap at 15 turns forces escalation** well before 50 messages; repeated
identical tool calls trip the **loop detector**; accumulated cost hits **$0.10 ceiling** → forced graceful escalation;
any abusive/injection content trips safety filter → block + escalate + flag account. Escalation enters the queue with the
**60 s SLA**; the human receives the summarised session state from Redis and the audit log. Conservative principle:
"better to escalate than to act incorrectly".

### Pre/Post handout extras
- None for W8S1 beyond the worksheet + reference solution.

### Likely doubts & exam angles
- **Q: Where are max-turn and cost ceiling enforced?** A: In the **agent runner** (hard limits: 15 turns, $0.10/session),
  not left to the LLM; guardrail layer forces graceful escalation.
- **Q: Why not let the LLM decide when to stop?** A: LLMs do not reliably self-terminate loops; hard limits are non-negotiable.
- **Q: Why a DistilBERT router before the LLM?** A: < 20 ms, ~250× cheaper than routing via a big LLM; order_status needs
  no LLM at all; saves $5+/hr.
- **Q: Why stateless pods + Redis?** A: 10K concurrent sessions, horizontal scaling, any pod can serve any turn; 100 MB of state fits easily.
- **Q: What is severity bias?** A: Hard cases escalate to humans regardless, making AI's resolved pool easier → inflated
  AI metrics; compare resolution per intent type.
- **Q: Escalation-rate guardrail band?** A: 5%–40%. Below → overconfident; above → too cautious.
- **Q: How is "resolved" defined?** A: No re-contact within 24h.
- **Q: How do you handle context overflow?** A: Sliding window of last N turns + summary of older turns; on overflow
  error, summarise oldest 50%; full history breaks around 20 turns.
- **Q: When not to use an agent?** A: Rule-based logic suffices, sub-100 ms latency needed, or high-stakes irreversible
  actions without human override.
- **Q: Why is A/B shorter (7 days) here than RAG (14 days)?** A: High-volume customer support → effects detectable faster;
  enterprise RAG has weekly usage cycles and lower volume.

---

## Bonus Case Study A — Ride-Hailing as an ML System (W8 "Session 3" bonus; not part of the course)
Source: w8_Ride_hailing_case_study_bonus.pdf. Trimmed excerpt covering **deep-dives 1–3 only** (Marketplace Forecasting,
Upfront Fare Prediction, Surge Pricing). Deep-dives 4–19 are omitted — "Use the ML Map and the ten-step rubric … to
design the remaining systems on your own. That repetition, not more reading, is what builds the skill."

### TL;DR
- **"One tap, sixty-three models"**: a Rapido rider in Bengaluru taps Confirm → ~400 ms later a bike is assigned, price
  shown, ETA promised. Ride-hailing = ML systems that must **cooperate under a shared latency budget while optimising
  objectives that actively harm each other**.
- Three properties that break your mental model: **two-sided conflict**, **A/B tests violate SUTVA**, **the system trains
  on its own decisions** (unserved demand is invisible).
- **Move models out of the request path**: surge and segment speeds are precomputed and read as lookups. "The question is
  never 'which model do we cut?' — it is 'which of these can be pre-computed?'"
- **Estimation selects the architecture**: 300K series forbids per-series models; 1.2 MB surge state forces a lookup.
- Fare: **bias, not variance, bankrupts you** → quantile regression, risk margin ρ(σ), daily bias reconciliation.
- Surge: logged price-demand gives **elasticity with the wrong sign** (confounded) → permanent 1–2% randomisation budget;
  **switchback** or geo randomisation instead of user-level A/B.

### Framing
**Why this domain:** a recommender has one customer; a fraud model has one objective. Ride-hailing has **two populations
with opposed interests**, a **physical constraint** (cars cannot teleport), and a **feedback loop** where today's dispatch
determines tomorrow's training data.

**Three properties that break your existing mental model**
1. **The two sides are in direct conflict.** Raise surge → drivers earn more, supply rises, rider conversion falls. Lower
   it → riders convert, drivers log off. No single metric to maximise; "the model improved" invites "**improved for whom?**"
2. **Your A/B test is statistically invalid.** 50% of riders in surge treatment book more cars → those cars unavailable to
   control riders → control contaminated. **Violates SUTVA**. Randomising users doesn't work in a marketplace.
3. **The system trains on its own decisions.** Dispatch sends drivers to the airport → data shows high airport volume →
   demand model learns airport is high-demand → more drivers sent. A neighbourhood never served records zero demand
   forever. "You are not observing demand; you are observing your own policy. **Unserved demand is invisible by construction.**"

**The latency budget is the real constraint** (~400–600 ms before the UI feels broken; "a model 3% better and 200 ms
slower is usually rejected"):
| Stage | Budget | What runs / why |
|---|---|---|
| Geocode + snap | 30 ms | Address → coordinates → road segment. Cached aggressively. |
| Fare + surge | 60 ms | **Surge is a lookup of a pre-computed grid, not a live model call.** |
| ETA (pickup + trip) | 120 ms | Largest slice. Routing engine + learned residual. |
| Fraud screen | 50 ms | Parallel with pricing. **Hard timeout returns "allow" — fail-open**: blocking a legit rider costs more than one fraudulent trip. |
| Dispatch match | 150 ms | Assignment over eligible pool; scales with pool size, so pool is pruned first. |
"Under time pressure the system lets suspected fraud through. Not a bug — a deliberate, quantified business decision.
**Knowing which way to fail is a system design skill.**"

### The ML Map (63 models, clusters A–H)
**A. Demand and Supply (6) — deep-dive 1:** 1 Spatio-temporal demand forecast (rides per hexagon per 15 min; H3/S2 grid;
horizons 5 min–7 days; regressors: holidays, weather, events, paydays) · 2 Supply forecast (within-session behaviour, not
headcount) · 3 Supply–demand imbalance prediction (feeds surge & repositioning; **forecasting the ratio directly beats
forecasting both sides separately**) · 4 Event & anomaly detection (step-change, must not fire on seasonality) · 5 Weather
effect modelling (rain raises demand and lowers supply — marketplace worst case) · 6 Long-horizon capacity planning
(driver recruitment per city per month; feeds finance).

**B. Marketplace Pricing (7) — deep-dives 2, 3, 4:** 7 Surge multiplier (per cell per minute) · 8 Price elasticity
(**must be causal** — observational data says high price ↔ high demand, backwards) · 9 Upfront fare prediction (asymmetric
loss — under-quote and company absorbs gap) · 10 Driver incentive optimisation (budget-constrained assignment over uplift;
pay only drivers whose behaviour changes) · 11 Rider promotion targeting (uplift segments: persuadables, sure things, lost
causes, sleeping dogs; discounting a sure thing = pure margin loss) · 12 Take-rate/commission optimisation · 13
Multi-product fare estimation (bike, auto, pool, XL, rental cannibalise).

**C. ETA and Routing (9) — deep-dives 5, 6, 7:** 14 Pickup ETA (judged most harshly) · 15 Trip ETA (routing baseline +
learned residual) · 16 Road-segment speed prediction ("your drivers are your traffic sensors") · 17 Map matching (HMM +
Viterbi, learned emissions) · 18 Route choice prediction ("ETA predicts reality, not policy") · 19 Pickup point
recommendation · 20 Destination prediction/autocomplete (get top-3 right → rider never types) · 21 Geocoding & POI
resolution (hardest for Gojek, Grab, Rapido — much of SEA/India lacks reliable addressing) · 22 Arrival detection &
geofencing (gates waiting charges → adversarially probed).

**D. Matching and Dispatch (17) — deep-dives 8–11:** 23 Driver–rider matching (bipartite assignment over batched window)
· 24 **Learned cost function inside the matcher** (optimiser is an LP; edge weights are ML predictions; model never
evaluated alone) · 25 Dispatch as sequential decision-making (RL/value functions; myopic vs long-run positioning) · 26
Batching window tuning (wait 5 s for better match vs assign instantly) · 27 Ride-pooling match · 28 Driver acceptance
prediction · 29 Idle-driver repositioning ("guidance, not instruction — contractors ignore bad advice") · 30 Cancellation
prediction · 31 Multi-product supply routing (rides/food/parcels — super-app).
**Driver-convenience block (32–39)** — each a feature drivers want and a constraint that shrinks the eligible pool: 32
Return-to-home/destination mode (capped 2 uses/day) · 33 Homeward-corridor demand forecasting · 34 Driver preference
modelling (trip-length bands, cash vs digital, AC, capacity, female-passenger-only matching — Uber India, Ola, Rapido) ·
35 Shift-end/go-offline prediction · 36 Break & rest compliance (**hard constraints, not penalties**) · 37 Fuel/charge-aware
dispatch (EV range; Delhi, Jakarta) · 38 Driver earnings-goal modelling (**backward-bending supply**) · 39 Heat-map &
guidance surfacing ("guidance nobody follows has zero effect").

**E. Trust, Safety and Fraud (10) — deep-dives 12–14:** 40 Payment fraud · 41 Incentive/promo abuse · 42 Collusion
detection (graph problem) · 43 GPS spoofing · 44 Account takeover & device fingerprinting · 45 Driver identity verification
(selfie vs licence) · 46 Crash & harsh-driving detection (telematics) · 47 Route deviation & in-trip anomaly · 48 Safety
incident triage (NLP over SOS; "highest-stakes error") · 49 Document verification (OCR + forgery).

**F. Content, Ranking and Discovery (5) — deep-dive 15:** 50 Home-screen vertical ranking · 51 Merchant/food
recommendation · 52 Search & autocomplete ranking · 53 Ads (pCTR, pCVR → auction) · 54 Notification send-time optimisation
("fatigue control beats targeting").

**G. Customer Experience and Operations (6) — deep-dives 16, 17:** 55 Support ticket classification & routing · 56 **LLM
support agent with tool use — "the W8S1 architecture, deployed on ride disputes"** · 57 Automated refund adjudication
(replay GPS trace vs planned route) · 58 Driver quality scoring (ratings sparse, biased, nearly all 5.0) · 59 Review &
content moderation (multilingual, Hinglish) · 60 Voice & IVR (core at Rapido, Gojek).

**H. Lifecycle and Growth (3) — deep-dives 11, 18, 19:** 61 Driver churn (driver acquisition cost dwarfs rider's —
highest-leverage retention model) · 62 Rider LTV & churn · 63 **Geo-experimentation & switchback testing** (not a model —
the measurement infrastructure every other model depends on).

**Trace one ride request — what fires and when:**
| Lane | What runs |
|---|---|
| PRE (scheduled, precomputed) | 1 demand · 3 imbalance · 7 surge grid · 16 segment speeds · 29 reposition |
| LIVE (request path) | 20/21 geocode → 19 pickup pt → 14/15 ETA → 9 fare → 40–44 fraud → 28/34 eligible → 23/24 match → assign |
| POST | 22 arrival · 47 deviation · 46 telematics · 58 rating · 42 collusion · 55–57 support |
"Surge (7) and segment speeds (16) are not called during the request — precomputed on a schedule and read from a KV store.
This is the most common way real systems buy latency: **move the model out of the request path and turn its output into a cache.**"

**Nineteen systems, sixty-three models** ("No company staffs a team per model"; grouping the way industry groups is part
of the skill):
| # | System | Models | Idea it teaches |
|---|---|---|---|
| 1 | Marketplace Forecasting | 1–6 | Censored demand; you observe your own supply |
| 2 | Upfront Fare Prediction | 9, 13 | Quantile regression; bias, not variance, bankrupts you |
| 3 | Surge Pricing | 7 | Causal elasticity; why logged price is confounded |
| 4 | Incentives and Promotions | 8, 10–12 | Uplift, not propensity; the knapsack allocator |
| 5 | Location Intelligence | 19–22 | The emerging-market core problem |
| 6 | Traffic and Map Inference | 16–18 | Your fleet is your sensor network |
| 7 | ETA Prediction | 14, 15 | Residual learning; asymmetric loss |
| 8 | Dispatch and Matching | 23–32, 34 | ML inside an optimiser; constraints cost efficiency |
| 9 | Ride Pooling | 27, 31 | Dynamic insertion; the first rider pays |
| 10 | Driver Positioning | 25, 29, 33, 39 | Herding; advice nobody must follow |
| 11 | Driver Lifecycle | 35, 36, 38, 61 | Multi-homing; backward-bending supply |
| 12 | Payment Fraud and Abuse | 40–42 | Extreme imbalance; graph signal; adversaries |
| 13 | In-Trip Safety | 43, 46–48 | A different cost function entirely |
| 14 | Identity and Documents | 44, 45, 49 | False positives take away income |
| 15 | Super-App Discovery | 50–54 | Cross-vertical cannibalisation of supply |
| 16 | Support and Disputes | 55–58 | Adjudication with an absent second party |
| 17 | Multilingual and Voice | 52, 59, 60 | Unequal model quality is unequal service |
| 18 | Rider Lifecycle | 26, 30, 62 | Cancellation as a churn event |
| 19 | Experimentation Platform | 63 | The infrastructure all of the above depend on |

**Reading guide:** Steps 1–3 — notice how the clarifying question ("requests or completed trips?", "who bears the risk?",
"who is the absent party?") determines the design. Step 5 — **the arithmetic selects the architecture**: 2 × 10^8 matching
pairs vs 150 ms forces candidate pruning; 1.2 MB of surge state forces a lookup; 300K forecast series forbids per-series
models; review capacity sets the fraud threshold. Step 8 — roughly **half the systems can use ordinary randomisation and
half cannot; the reason is always shared scarce supply.** Step 9 — fallback ladder and which way the component fails.
Step 10 — "where the judgement lives". **If you read only three:** deep-dive 3 (elasticity's wrong sign), 8 (ML as
coefficients inside an optimiser; cost of return-to-home), 19 (why a marketplace can't use the standard A/B).

### Deep-Dive 1 — Marketplace Forecasting (models 1–6)
**Why one system:** shared spatial grid, time buckets, event/weather regressors, backfill machinery — and downstream
consumes **their ratio**. Building separately → inconsistent geometry that cannot be divided.

**Step 1 — Clarify.** What consumes the forecast (surge, repositioning, incentives, quarterly hiring — different horizons
and error tolerances)? Spatial resolution? Cost of over- vs under-forecasting supply? **Requests or completed trips?**
- **The clarification that determines everything:** logs record completed trips = **min(demand, supply)** — *censored* by
  supply. In an under-supplied cell they tell you how many drivers were there, not how many riders wanted one. Train on
  completed trips → model forecasts your own historical supply → used to allocate supply → **shortage becomes permanent
  and invisible**. Must recover true demand from app opens, searches, timed-out requests, and requests abandoned after
  seeing ETA.

**Steps 2–3.** Forecasting is upstream of nearly everything (surge reads imbalance; repositioning reads the forecast
gradient; incentive budgets days ahead; recruitment quarterly). Errors surface as bad pricing, misdirected drivers, wasted
incentive spend — not as "forecast complaints". **Problem:** for each H3 cell h and future bucket t produce (i) expected
requests D̂(h,t) corrected for censoring, (ii) expected available driver-hours Ŝ(h,t), (iii) predictive distribution of their
ratio — horizons 5 min to 90 days; refreshed every 5 min (short) and daily (long).

**Step 4 — Requirements.**
| Functional | |
|---|---|
| Multi-horizon | One coherent system 5 min (dispatch, surge) → 90 days (hiring); horizons must not contradict |
| Multi-resolution | Cell, zone, city — reconciled so cells sum to city |
| Censoring correction | Recover latent demand from supply-constrained observations |
| Distributional output | Surge/incentives need uncertainty; high-variance cell ≠ confidently busy cell |
| Event awareness | Calendar of concerts, matches, festivals, flight schedules |
| Backfill | Recompute history when road graph, grid or definitions change |
| **Non-functional** | |
| Freshness | Short-horizon forecast ≤ 5 min old |
| Serving | **A lookup, never a live model call** |
| Scale | ~150K active cells × multiple horizons |
| Graceful degradation | Fallback to **seasonal-naive** (same cell, same hour, last week), always available |

**Step 5 — Estimations.**
- Series: 150K cells × 2 (demand, supply) = **300K series**; 15-min buckets = 96 pts/day/series → **~29M obs/day**.
- **Rules out per-series models** (300K ARIMA/Prophet artefacts unmonitorable) → **single global model with cell identity
  as a feature (learned embedding)**.
- **Sparsity:** most cells < 5 rides/bucket; many zeros → **count data with excess zeros, not Gaussian regression** (most
  common modelling mistake).
- Training data: 2 years × 29M/day ≈ **21B rows** → aggregate and sample.
- Served forecast storage: 300K × 20 horizons × 3 quantiles × 4 bytes ≈ **72 MB** — cacheable in memory at every edge.

**Step 6 — Architecture.**
- **(a) Data Collection:** demand-side signal chain > trip table: app opens → destination searches → fare-quote views →
  requests → requests with no driver → requests abandoned after price/ETA; drop-offs reconstruct unmet demand. Supply:
  driver online/offline, pings, status (idle/en route/on trip). Event calendar, weather feed, **regional holiday calendar**
  (India's holidays vary by state).
- **(b) Data Engineering:** streaming path aggregates into H3 cell × 5-min buckets (short horizon); batch path builds
  training table with **point-in-time correctness**. Classic bug: joining observed weather instead of the forecast
  available at the time — "**Train on the forecast you would have had, not the truth you learned later.**"
- **(c) Feature Engineering:** Temporal — hour-of-week cyclic (sin/cos, so 23:00 and 00:00 adjacent), holiday flags,
  **days-to-payday** (strong in cash economies). Lags — t−1, t−96 (yesterday), t−672 (last week); rolling means. Spatial —
  same lags for **H3 k-ring** neighbours. Static cell — land use, POI density, transit proximity, historical mean/var.
  Exogenous — weather forecast, event proximity, flight arrivals for airport cells.
- **(d) Preprocessing:** **censoring correction** (hard part): where supply was binding (near-zero idle drivers, high
  timeout rate), observed count is a *lower bound*. Options: model only uncensored buckets and extrapolate; **Tobit-style
  censored likelihood**; use the funnel (searches → requests) to estimate conversion. Aggregate sparse cells upward; flag
  anomalous periods (outage, lockdown) so they don't poison seasonality.
- **(e) Models:** Baseline hard to beat: **seasonal-naive**; report every model against it. Workhorse: **single global
  gradient-boosted model** with cell identity as categorical + lags (cheap, robust, debuggable). Neural: global deep model
  (temporal-conv / transformer) learning cell embeddings — helps sparse cells. **Count-aware loss:** Poisson or
  negative-binomial, or quantile loss; squared error on sparse counts systematically over-predicts quiet cells.
- **(f) Post-processing:** **hierarchical reconciliation** (MinT or proportional) so cells sum to zone/city (else incentive
  budgets over/under-spend); clip negatives; event overrides; write to forecast store.

**Step 7 — Evaluation.**
| Metric | Why / trap |
|---|---|
| WAPE | Preferred over MAPE for counts — MAPE undefined/explosive at zero actuals |
| Pinball loss | Right metric for quantile outputs; evaluates the whole distribution |
| Bias by cell and hour | Systematic error worse than noise — downstream allocation compounds it |
| **Skill vs seasonal-naive** | "The only honest framing." "WAPE 18%" meaningless; "12% better than seasonal-naive" is a result |
| Accuracy at the peaks | Evaluate top-decile-demand buckets separately — the ones the business acts on |
Online: measure through consumers — did surge over-trigger, did guidance send drivers to quiet cells, did incentive spend
land where demand materialised? **"Guidance regret":** drivers who followed the forecast earned less than those who ignored it.

**Step 8 — A/B.** Can't A/B a forecast (everyone shares one future); test the decisions it drives → interference →
switchback. Cheap & valid: **backtesting with rolling origins** (valid because a forecast, unlike a policy, doesn't change
the world it predicts); guard against leakage — only features available at origin.

**Step 9 — Deployment.** Forecasts are **written, not served** (scheduled job → store; request path reads). Monitor
forecast-vs-actual (automatic delayed label), bias drift per city, feature freshness, fraction of cells on naive fallback,
reconciliation residuals. **Alert on structural breaks** (new metro line, road closure, competitor promo). Retrain weekly;
refit seasonal components less often.

**Step 10 — Risks.**
- **Self-fulfilling forecast:** forecast low → send no drivers → riders fail and stop opening app → observed demand falls →
  forecast "confirmed"; neighbourhoods quietly written out of service while dashboards show excellent accuracy.
  Mitigations: forecast requests and app-opens, not completed trips; deliberate supply coverage in low-forecast areas;
  monitor unmet-demand and app-open-without-request rates per cell as a **first-class equity metric**.
- **Black-swan periods poison training:** flag anomalous periods as features (don't silently drop), human override path;
  requirement: **fail visibly rather than confidently**.
- **Accuracy vs actionability:** ±2 on 3 is useless; ±50 on 800 is actionable. Weight evaluation by the decisions driven.

### Deep-Dive 2 — Upfront Fare Prediction (models 9 & 13)
**Why not the same as ETA:** once a fixed price is quoted the company absorbs the difference — "**every ETA error becomes a
P&L entry**". Prediction → risk-pricing problem; same team as ETA, different loss function.

**Step 1 — Clarify.** Binding quote or estimate reconciled to metered? Who absorbs the gap (platform, driver, rider)? Is
the driver paid on quoted or actual time (determines incentive to take a longer route)? Mid-trip destination change?
Per-product quotes / cannibalisation?
- **Critical question: who bears the risk?** Rider pays quote + driver paid on actuals → platform absorbs variance (needs
  accurate unbiased model + reserve). Driver paid quote-derived → driver absorbs it (cheaper, deeply unpopular). Most keep
  risk centrally **"because the platform can pool it across millions of trips while an individual driver cannot. That
  pooling argument is the answer an interviewer is looking for."**

**Steps 2–3.** Metered fares create anxiety and suppress booking; fixed price raises conversion. "The platform is selling
certainty and charging for it; the cost of providing that certainty is the variance of its own prediction errors."
**Problem:** given origin, destination, product, time, conditions → price s.t. expected realised cost covered, quote
competitive enough to convert, and aggregate quoted-minus-actual gap is a **controlled, budgeted number** — within the
**60 ms** pricing slice.

**Step 4 — Requirements.**
| Functional | Detail |
|---|---|
| Multi-product quoting | Bike, auto, car, XL, pool, rental simultaneously, mutually consistent (XL never cheaper than bike) |
| Cost-side prediction | Predict duration & distance **distributions**, not points |
| Risk margin | Explicit, tunable buffer separate from point prediction |
| Re-quote rules | Destination change, long waits, extra stops |
| Price consistency | Same trip quoted twice within seconds must not differ visibly |
| **Non-functional** | |
| Latency | < 60 ms for all products together |
| Aggregate bias | mean |quoted − actual| within stated tolerance of zero, monitored daily |
| Auditability | Every quote reconstructible for disputes |

**Step 5 — Estimations.**
- Quote volume: browse = 4× booking rate, 6 products/view, 3M bookings/day → **~72M product-quotes/day, ~5K QPS peak**.
  "Design for the browse path, not the purchase path."
- **Economics of being wrong:** MAE Rs. 8 on Rs. 150 symmetric → nets out. **2% systematic under-quote = 3M × Rs. 3 =
  Rs. 9M/day ≈ Rs. 3.3B/year. "Bias, not variance, is what bankrupts you."**
- **Tail exposure:** 1% of trips take > 2× predicted time; at Rs. 200 extra each → 30K × Rs. 200 = **Rs. 6M/day** —
  reserve line; reason the model must output a distribution.

**Step 6 — Architecture. Fare is a composition:**
```
Fare = base(product) + r_d · d̂ (distance) + r_t · T̂ (duration)  ×  m (surge lookup)  +  ρ(σ) (risk margin)  + tolls, fees
```
- **(a) Data:** completed trips with actual duration, distance, waiting, tolls; quote shown; whether rider converted at
  that quote (**elasticity signal**); reconciliation outcome. **Log quotes that were not accepted** — the only evidence of
  price sensitivity, routinely dropped to save volume.
- **(b) Data Eng:** **shares the ETA feature pipeline entirely** — "build a second consumer of the same features with a
  different head and loss." A parallel pipeline guarantees fare and displayed ETA eventually disagree.
- **(c) Features:** everything ETA uses + toll-road probability, waiting-time expectation (traffic-light density,
  pickup-point difficulty), **destination ambiguity** (vague destination → higher variance → larger margin), product
  cross-features.
- **(d) Preprocessing:** exclude mid-trip destination changes (handled by re-quote rule). **Do NOT exclude the long tail** —
  unlike ETA winsorising, for fare the tail *is* the financial exposure; trimmed data calibrates the margin "to a world
  without bad days".
- **(e) Models:** **quantile regression** — predict P50, P80, P95 of duration; **GBTs with pinball loss**. Median → point
  estimate; spread between quantiles = σ → risk margin. "One model gives you both the price and the confidence to charge
  for, with no separate uncertainty model."
- **(f) Post-processing:** apply surge as lookup (never recompute); add ρ(σ); **enforce cross-product monotonicity** (XL
  below bike = immediate exploit); round to rider-legible values; **cache quote briefly** so refresh doesn't change it.

**Step 7 — Evaluation.**
| Metric | Why |
|---|---|
| **Aggregate quote bias** | The number that matters: mean(quoted − actual) daily per city; within band around zero |
| Pinball loss by quantile | Are quantiles honest? If P95 is exceeded 12% of the time, margin too thin |
| Tail loss (p99 gap) | Your reserve requirement |
| Conversion at quote | Accurate but uncompetitive quote loses the trip |
| Per-segment bias | Aggregate can be zero while losing on bikes and gaining on cars |
Online: realised margin per trip, fare-dispute/refund rate, conversion by product, re-quote frequency.

**Step 8 — A/B.** Rider-level randomisation **mostly valid** (quoting one rider differently doesn't consume shared supply).
Cautions: (i) systematically lower quotes → treated riders book more and draw supply → interference at large traffic
shares; (ii) fairness/legal — "**randomise the model, not personalised prices**"; document it as a bounded experiment on
estimation accuracy, not willingness-to-pay. Guardrails: conversion, realised margin, dispute rate.

**Step 9 — Deployment.** **Shadow-quote for a week** vs metered actuals before any rider sees it. **Daily bias
reconciliation** (sum quoted − actual by city & product; alert outside band) — "a finance-grade control". **Fallback:
model unavailable → metered pricing, tell rider fare is estimated. Never guess a binding price without a model** (open-ended
liability). **Circuit breaker on aggregate bias** → auto-revert to previous model version.

**Step 10 — Risks.**
- **Adverse selection on destination ambiguity:** vague-destination riders underpriced and subsidised by everyone; grows
  over time; ρ(σ) exists to price it.
- **Driver incentive misalignment:** driver paid from quote → optimises for speed (good). Driver paid on actual time while
  rider pays fixed → driver benefits from slower trips; platform pays. "**A pricing scheme that pays people to waste time
  will reliably produce wasted time.**"
- **Certainty vs fairness in the tail:** fast-trip riders overpay, funding the pool — insurance; must be explained
  honestly; margin must be stated and bounded, not "quietly growing" (loses regulatory goodwill).

### Deep-Dive 3 — Surge Pricing (model 7)
**Step 1 — Reject the common framing.** "Surge is not a revenue-extraction model." It solves an **allocation problem**:
when 100 riders want 40 cars, who gets one and by what mechanism? Without price → queue (everyone waits; drivers have no
reason to move). With price → demand rationed to those who value it most, supply pulled toward shortage. "**The output is
not a price. The output is a reallocation of cars across the city.** If you frame surge as pricing, you will build the
wrong metrics."
Questions: regulatory cap? (India 1.5–2× caps; Indonesia mandates fare floors & ceilings.) Price shown before or after
booking? Surge rider price, driver payout, or both — may they differ? Smallest pricing unit? How often may the multiplier
change without feeling erratic?

**Steps 2–3.** Most politically/reputationally sensitive system; primary short-term balance lever — one of only **two tools
(surge, repositioning guidance)** that act in the next 10 minutes. **Problem:** for each cell and minute choose m ∈ [1.0,
m_max] maximising a weighted combination of completed trips, rider surplus and driver earnings, subject to regulatory caps
and a volatility constraint — given demand forecast, supply forecast, and **how each side responds to price** ("that last
term is the entire difficulty").

**Step 4 — Requirements.**
| Functional | |
|---|---|
| Per-cell pricing | Multiplier per H3 cell per minute, every city |
| Spatial smoothing | Adjacent cells must not differ wildly (walk 50 m to save 40% → gamed within days) |
| Temporal smoothing | Rate-limit changes (1.0/2.0/1.0 oscillation destroys trust) |
| Regulatory caps | Hard constraint, auditable |
| Emergency suppression | Auto cap/disable in disasters — **must not depend on a human noticing** |
| Rider transparency | Show multiplier before booking |
| **Non-functional** | |
| Latency | **< 10 ms — a lookup, not a model call** |
| Update cadence | Recompute every 30–60 s per cell |
| Auditability | Every multiplier reconstructible months later |
| Blast radius | A bad deploy must not price an entire country at 5× |

**Step 5 — Estimations.**
- Cells: mid-size Indian city ≈ 1,500 active H3 res-8 cells (~0.7 km² each); 100 cities → **~150K cells**.
- Compute: 150K cells every 30 s = **5,000 computations/s** — trivially small; output a tiny (cell, multiplier) table.
- Storage: 150K × 8 bytes ≈ **1.2 MB hot state** — "fits in a CPU cache", replicated to every edge, in-memory lookup.
- Audit log: 150K × 2,880 updates/day × ~200 bytes ≈ **86 GB/day** — logging is **three orders of magnitude** more expensive
  than computation, and mandatory.
- "A junior design puts a 'surge model' in the request path. The numbers say it should never be there."

**Step 6 — Architecture.**
- **The mistake everyone makes first:** m = clip((D̂/Ŝ)^α, 1.0, m_max). A real baseline most companies ship first; wrong
  because it assumes a fixed imbalance→price relationship, when what matters is **how much demand falls and supply rises per
  unit price** — varies by city, hour, cell, segment (1.5× clears an airport at midnight; does nothing in a business
  district at 6 pm where riders expense fares).
- **Why you cannot learn elasticity from historical data:** logs show high multiplier ↔ high demand; naive model concludes
  raising prices increases demand. Causality reversed — price was set by a policy that looked at demand → confounded →
  **elasticity estimate with the wrong sign.** Generalises: "**any time your feature was set by a policy that also saw the
  outcome, supervised learning gives you a correlation you must not act on.**"
- **Getting causal elasticity:**
  | Approach | How | Cost |
  |---|---|---|
  | Deliberate randomisation | Small random jitter to multiplier in random cells/times — breaks confound | Gold standard; costs real revenue; must be small & bounded |
  | Switchback experiments | Flip whole cell between policies on a time schedule | The workhorse; careful window design |
  | Natural experiments | Rollout boundaries, regulatory changes, cap discontinuities | Free, rare, hard to defend |
  | Regression discontinuity | Cells just above/below a multiplier rounding threshold | Clever, cheap, local to threshold |
  **Practical answer: permanent small randomisation budget — ~1–2% of cell-minutes jittered, forever.** Teams that skip it
  discover their pricing model was mis-calibrated for a year.
- **(a) Data:** request events incl. **non-converting requests** (elasticity signal, easy to forget), driver on/offline,
  positions, completed trips, multiplier shown, saw-and-abandoned; **log counterfactual context** (what baseline policy
  would have shown).
- **(b) Data Eng:** streaming per H3 cell per minute (open requests, available drivers, in-progress trips, recent
  conversion) → hot state store; parallel warehouse path for elasticity estimation & audit.
- **(c) Features:** imbalance (forecast D, S, ratio, trends); elasticity priors per cell/hour cluster; context (weather,
  events, day type, flight bank landed); neighbourhood imbalance ("a shortage next door is about to become your shortage").
- **(d) Preprocessing:** **cluster cells into elasticity groups**; hierarchical estimate shrinking sparse cells toward group mean.
- **(e) Models — three components, not one:** (i) demand & supply forecasts (supervised time series); (ii) elasticity
  estimates (causal, from randomisation budget); (iii) **an optimiser, not a model**, choosing m under caps & smoothing.
  "The ML produces inputs; a constrained optimisation produces the decision."
- **(f) Post-processing:** spatial smoothing; temporal rate-limiting; hard clip at regulatory cap; emergency override;
  rounding; write to hot store — "the request path only ever reads".
- **Supply side bends backwards:** (1) **slow and physical** — drivers take 5–15 min to reach surge cell; shortage may be
  gone → surplus, resentment, oscillation; (2) **backward-bending** (model 38) — earnings-target drivers log off sooner when
  pay rises; above some multiplier, raising price can *reduce* supply hours; (3) **anticipatory and gameable** — drivers
  collectively go offline to manufacture surge; "a repeated game against a strategic counterparty".

**Step 7 — Evaluation.** Offline evaluation of a pricing policy is genuinely hard — "say so out loud in an interview";
counterfactual unobservable. Can do: component forecast accuracy; elasticity CI tightness; **agent-based marketplace
simulation** (directional only — "a policy that wins in simulation has proved it beats your beliefs, not reality");
**constraint satisfaction** checks (cap breaches, rate-limit oscillation, spatial cliffs — the ones that make news).
| Online metric | Whose interest |
|---|---|
| **Completed trips** | **North star.** Revenue/trip ↑ while trips ↓ = failure |
| Request fulfilment rate | Rider |
| Rider surplus proxy | Rider — conversion per price point; abandonment after seeing multiplier |
| Driver earnings/hour | Driver — determines whether drivers stay |
| Driver utilisation | Driver — paid fraction of online time; summoning to resolved shortage destroys it |
| ETA to pickup | Both — the physical outcome surge should improve |
| Surge incidence | **Guardrail** — % requests with m > 1; steady rise = revenue tool, regulators notice |
"**You must report both sides, always.**" +8% driver earnings, −5% completed trips is a business decision, not the model
owner's; presenting only the improved metric is how ML teams mislead leadership.

**Step 8 — A/B: where the standard method breaks** ("methodological heart of the session").
- **Rider-level randomisation is invalid:** treated riders (lower multiplier) book more, take the cars; control faces
  fewer drivers/longer ETAs caused by treatment → effect **inflated** (real effect + harm to control); "ship this and the
  effect vanishes at 100% rollout".
- **SUTVA** (Stable Unit Treatment Value Assumption): one unit's treatment doesn't affect another's outcome — violated by
  construction with shared scarce supply. **Fix: randomise at a unit that contains the interference.**
- **Switchback randomisation (standard fix):** switch whole city/cell-cluster between A and B every 30–60 min. Unit of
  analysis = **time period**. Periods long enough for **carryover** to decay, short enough for power. **Randomise order**
  (not strict alternation — weekly patterns confound). **Cluster standard errors by period** — treating trips as
  independent grossly overstates significance (most common analysis error).
- **Geo/cluster randomisation (alternative):** whole cities or separated clusters; handles long carryover; but few,
  heterogeneous units → poor power → **difference-in-differences or synthetic control**; boundaries leak → **buffer zones**.
- **Carry beyond ride-hailing:** interference in ad auctions, marketplaces, social feeds, delivery platforms. Ask "**do my
  treatment and control units compete for anything?**" before every experiment.

**Step 9 — Deployment.** Staged rollout **by city, never globally**. **Hard circuit breakers independent of the model**:
un-exceedable ceiling; auto kill-switch if fraction of surging cells exceeds threshold; alert if any cell's multiplier
changes more than a delta in one step. **Emergency detection** auto-suppresses surge ("build it before you need it").
Monitor: multiplier distribution per city, surge incidence trend, conversion per multiplier band, driver repositioning
response, **cap-breach attempts** (optimiser wanting to exceed cap = genuinely unmet demand), spatial-gradient violations.
Audit every multiplier with inputs — "'the model decided' is not an acceptable answer."

**Step 10 — Risks.**
- **Surge during emergencies is an ethics problem, not a modelling problem:** automatic suppression on pre-defined signals
  (demand anomaly + external feeds). "Models optimise the objective you write down. If the objective says nothing about
  human dignity in a crisis, the model will not supply it."
- **Fairness & disparate impact:** persistent surge in low-income, thin-supply areas; correlates with geography tied to
  caste/religion/race; monitor multiplier distribution by area demographics; be prepared to subsidise underserved areas.
- **Regulatory caps change the optimal design:** under 1.5× cap price can't clear peak → non-price mechanisms: queueing,
  explicit wait estimate, demand shaping ("try again in 10 minutes"), driver incentives from platform margin, honest
  unavailability. "Candidates who simply add min(m, 1.5) have missed the question."
- **Trade-off table:** No surge (simple, popular, fair-seeming; long waits, unmet demand, drivers stay home) · Aggressive
  surge (market clears, drivers earn; rider anger, regulatory risk, equity) · **Capped surge** (common compromise; needs
  explicit rationing machinery). "There is no correct answer, and saying so with the trade-offs named is a stronger
  interview answer than picking one confidently."

### Likely doubts & exam angles (ride-hailing)
- **Q: Why is fraud fail-open?** A: Under the 50 ms timeout, blocking a legit rider costs more than one fraudulent trip — a quantified business decision.
- **Q: Why train on requests/app-opens rather than completed trips?** A: Completed = min(demand, supply) — censored; trains a supply forecast and makes shortages permanent.
- **Q: Why a single global model instead of per-cell models?** A: 300K series unmanageable; global GBT with cell embedding/categorical; shares structure for sparse cells.
- **Q: Why not MAPE?** A: Undefined/explosive at zero actuals; use WAPE, pinball loss, skill vs seasonal-naive.
- **Q: Why quantile regression for fares?** A: Need distribution: median for price, quantile spread σ for risk margin; bias drives losses (2% = Rs. 3.3B/yr).
- **Q: Why is surge a lookup?** A: 5K computations/s producing 1.2 MB of state — precompute every 30–60 s; < 10 ms read.
- **Q: Why does logged data give the wrong elasticity sign?** A: Price set by a policy reacting to demand → confounded; use randomisation (1–2% jitter), switchbacks, natural experiments, RDD.
- **Q: Why can't you A/B surge by user?** A: SUTVA violation via shared supply; use switchback (cluster SEs by period) or geo randomisation (DiD/synthetic control, buffer zones).
- **Q: Can upfront fare be A/B'd per rider?** A: Mostly yes (no shared supply consumed), with cautions about interference at large share and price-fairness — randomise the model, not prices.
- **Q: What is backward-bending supply?** A: Earnings-target drivers log off sooner when pay rises, so higher surge can reduce supply hours.

---

## Bonus Case Study B — Designing a Coding Agent at Scale (W8 "Session 4" bonus; not part of the course)
Source: w8_Coding_agent_case_study_bonus.pdf. Trimmed excerpt covering **deep-dive 1 only (Cache Economics)**. Omitted
deep-dives 2–5: Context Assembly Under a Token Budget, The Agent Loop and Control, Evaluation Without Ground Truth,
Prompt Injection in an Agentic Loop.

### TL;DR
- **"One instruction, sixty-three model calls"**: "the auth test is failing, fix it" → ~40 s, **~63 model calls, ~900K
  input tokens**.
- A coding agent is the clearest case where **"the cost function and the product are the same object"** — every quality
  improvement costs tokens; tokens are the dominant opex. Business metric = **cost per resolved task**, not cost per call.
- Resending history makes total tokens grow **~N²** with step count: doubling steps ≈ quadruples the bill.
- **Prefix caching** (write ≈ 1.25×, read ≈ 0.1×) breaks even at **n ≥ 2** and saves **~86%** at n = 30 → "**agent economics
  are decided by prompt layout, not model selection**". **Order the prompt by volatility.**
- One volatile token at the head (timestamp, turn counter, shuffled tool list) silently drops hit rate to ~0 — invisible
  to everything except a cost dashboard. Hash the prefix in CI; cache hit rate as a first-class SLO; namespace cache keys by tenant.

### Framing
**Why this domain rather than a chatbot:** a chatbot issues one call per turn against user-typed context; a coding agent
issues tens to hundreds of calls per instruction against a context that grows with every tool result, over a codebase
larger than any window; it **acts on the world** (edits files, runs commands), **reads untrusted text**, and its outputs
become training data for its successor. Inference efficiency, retrieval, agent control and monitoring all become
load-bearing at once.

**Three properties that break the chatbot mental model**
1. **The cost function is the product.** An hour-long task consumes millions of tokens; metric = cost per resolved task;
   it fights quality directly (more exploration/verification raise both resolution rate and bill). A larger model doesn't
   escape — same token volume × higher price.
2. **The system cannot be evaluated directly.** No label; must execute code, and a passing suite is weak evidence (agent
   may weaken an assertion, special-case the input, fix a symptom). Public benchmarks **saturating, plausibly contaminated**,
   correlate imperfectly with private codebases — a sharper offline-vs-online gap since "the offline metric is itself contested".
3. **The system trains on trajectories it generated.** Accepted diffs → preference data, but outcomes only observed for
   paths chosen (policy fingerprint) and acceptance is an unreliable proxy (engineers approve code they haven't read).
   "Today's policy determines tomorrow's training distribution."

**The token budget is the real constraint — why cost grows with the square of step count** (one 30-step task, naive
design resending full history):
| Stage | Input tok | Cumulative | Driver |
|---|---|---|---|
| System prompt + tools | 12,000 | 12,000 | Fixed, identical every call — what makes it cacheable |
| Step 1, first search | 14,000 | 14,000 | Prompt + user instruction |
| Step 5 | 38,000 | 130,000 | Four file reads + two search results in history |
| Step 15 | 94,000 | 830,000 | Test log + several diffs accumulated |
| Step 30 | 180,000 | 2,600,000 | Near window limit; compaction forced |
"Read the cumulative column." Total ~N². Prefix caching makes repeats cheap; **compaction bounds growth; step ceilings
exist because a non-converging agent is unboundedly expensive.** "A design that ignores the quadratic term will be correct
and unaffordable."

**What it costs and what it rules out** (illustrative, dated Sept 2026; "ratios matter more than absolute numbers"):
- **Uncached:** 2.6M input tokens × ~$3/M ≈ **$7.80/task**. 200 engineers × 10 tasks/day ≈ **$15,600/day ≈ $4.7M/year** —
  naive design never shipped.
- **With prefix caching:** repeated prefix charged once at write premium then ~1/10 read price → task ≈ **$1.00–1.50**.
- **The lever is layout, not model choice:** cheaper model lowers unit price but not quadratic growth, and usually raises
  step count (worse). Caching attacks the growth itself.
- **Step ceilings bound the tail:** a non-converging task is the most expensive item of the day; a ceiling converts an
  unbounded loss into a bounded one.
- "**Estimation is not a ritual performed after the design is chosen; it is what chooses the design.**"

### The ML Map (65 subsystems, clusters A–G; ⋆ = deep-dived in full doc)
**Why five deep-dives:** each introduces a structurally different problem — cache economics (arithmetic selects
architecture), context assembly (ML supplies coefficients to a classical optimiser), agent loop (control under uncertainty
about own progress), evaluation (no ground truth), injection (adversary adapts). Most of the other 60 are variations:
speculative decoding (14) and tiered routing (17) = cache economics with different arithmetic; code search ranking (39) =
context assembly; supply-chain checking (57) = adversarial pattern. "The transferable skill is recognising which of the
five a new subsystem resembles."

- **A. Context & Retrieval (1–10) — "what the model is permitted to see":** 1 Repo map (symbols/signatures without bodies)
  · 2 AST-aware chunking ("a chunk ending mid-function teaches the model a syntax error") · 3⋆ Agentic search vs embedding
  retrieval · 4 Embedding index maintenance (agent invalidates index by editing) · 5⋆ Context compaction/summarisation ·
  6⋆ Cache-aware prompt ordering · 7 Stale-context invalidation (file read at step 3, edited at step 9 — tell model which
  is current) · 8 Multi-file edit coherence · 9 Dependency-graph traversal (blast radius) · 10 History pruning policy.
- **B. Inference Serving (11–22) — "the cost layer":** 11⋆ **Prefix cache hit-rate engineering — single largest cost
  lever** · 12 KV cache memory mgmt (block-based allocation) · 13 Continuous batching (throughput ↑, per-user latency ↑) ·
  14 Speculative decoding (draft proposes, target verifies; very effective on predictable code) · 15 Disaggregated prefill
  & decode (prefill compute-bound, decode bandwidth-bound; agent workloads prefill-heavy) · 16 **Cache-aware routing**
  (round-robin destroys locality) · 17 Tiered model router (cheap tier mechanical edits, frontier tier planning; routing is
  itself a prediction problem) · 18 Constrained decoding (JSON schema at sampling time) · 19 Long-context attention cost
  (final steps most expensive per token) · 20 Session affinity & eviction · 21 Autoscaling for bursty agent traffic · 22⋆
  Cache write-cost accounting.
- **C. Agent Loop & Control (23–32):** 23 Tool-call planning · 24 Subagent spawning (fresh context, return summary — a
  context-management technique) · 25⋆ Loop & oscillation detection · 26⋆ Budget enforcement (token, wall-clock, step
  ceilings) · 27 Interrupt & steering · 28 Checkpoint & rollback · 29 Parallel tool execution (reads concurrent, writes
  serialise) · 30⋆ Permission gating · 31 Retry & error recovery · 32⋆ Termination & surrender detection ("honest partial
  failure > unsubstantiated success").
- **D. Code-Specific Models (33–40):** 33 FIM completion vs chat-style edit · 34 Diff application & patch repair · 35
  Next-edit prediction · 36 Test generation & test-quality scoring ("a test that cannot fail is worse than no test") · 37
  Lint/type-error repair loops ("compiler is a free and exact verifier — cheapest quality gain") · 38 Commit/PR description
  generation · 39 Code search ranking (symbol relevance) · 40 Language-specific tokenisation (indentation-sensitive languages).
- **E. Evaluation (41–50) — "the layer with no ground truth":** 41⋆ LLM-as-judge with bias controls · 42⋆ Execution-based
  harness · 43 Contamination detection · 44 Per-language/framework slicing · 45 Regression suite maintenance (frozen,
  rotated) · 46 Human preference collection (low inter-annotator agreement) · 47 **Accept rate as online proxy — biased
  upward** · 48 **Revert rate & follow-up-turn rate — lagging, more honest** · 49 Trajectory replay for counterfactual
  comparison · 50 Eval-set hygiene & holdout rotation.
- **F. Safety & Trust (51–58) — "the counterparty adapts":** 51⋆ Prompt injection detection in retrieved content · 52
  Trust-boundary/provenance tracking · 53 Secret exfiltration prevention · 54 Destructive-command gating · 55 Sandbox
  hardening · 56 Network egress default-deny · 57 Supply-chain-aware dependency suggestion (typosquatting) · 58 Licence checking.
- **G. Training Flywheel (59–65):** 59 Trajectory collection (retention policy is a modelling decision) · 60 Preference
  pair construction · 61 Rejection sampling · 62 **RLVR with execution rewards** (tests passing = verifiable reward;
  coding has clean RL signal) · 63 Distillation to cheaper tiers · 64 Eval-set contamination firewall · 65 Policy-confounding
  correction (importance weighting or permanent randomised exploration budget).

**Trace one agent task — what runs once, what runs thirty times:**
| Lane | Steps |
|---|---|
| PREWARM (once, amortised) | 1 repo map · 11 prefix cache · 4 embed index · 18 tool schemas · 21 warm replica |
| LOOP ×30 (dominates bill) | 6 assemble → 17 route → 12–16 model call → 23 parse tool → 30/51 gate → 29/55 execute → 5/25 observe → done? (else iterate) |
| POST (async, cheaper model) | 42 execute tests · 41 judge · 59 log trace · 47/48 accept/revert · 60 pref pairs · 62 RLVR |
"**Move work out of the hot lane.** … not which step to cut, but which can be hoisted into PREWARM or deferred to POST."
(note: same principle as ride-hailing's "move the model out of the request path".)

### Deep-Dive 1 — Cache Economics (subsystems 11 & 22), full rubric
**Step 1 — Clarify.** Objective = **cost per resolved task, not cost per call** (a cache strategy that lowers per-call
cost but raises step count made the system worse). Questions: cache lifetime — does it span a session? Write premium and
ratio? Can a session be pinned to the replica holding its prefix? Is the tool schema stable per session?
- **Reframe:** looks like infra (delegated to platform team) but is a **prompt-design decision** — the cache boundary is
  set by where volatile content sits in the prompt.

**Steps 2–3.** ~30 calls/task, each resending accumulated history → large, largely identical leading block. **Problem:**
minimise total token spend across a task at a fixed quality bar by maximising fraction served from cache; decision
variables = **ordering of prompt segments + lifetime policy**; constraint: needed content stays present and current.

**Step 4 — Requirements.**
| Functional | |
|---|---|
| Stable prefix | System prompt, tool schemas, repo map **byte-identical** for a session |
| Correct invalidation | Drop cached content when file or schema changes |
| Prefix-aware routing | Reach the replica holding the prefix |
| Cost attribution | Tokens & spend per session, user, deploy |
| **Non-functional** | |
| Cache hit rate | **> 90% on step 2 onward**; below that investigate before shipping |
| TTFT | **< 2 s at p95** (a miss shows up as latency) |
| Observability | Hit rate per deploy, alert on drop |
| Isolation | **No cache content shared across tenants under any circumstance** |

**Step 5 — Break-even calculation** (write ≈ 1.25×, read ≈ 0.1× of normal input price):
- Uncached: prefix P read on n calls = **nP**.
- Cached: **1.25P + 0.1P(n − 1)**.
- Break-even: 1.25 + 0.1(n − 1) < n holds for **n ≥ 2** → "**Cache anything read more than once.**"
- At n = 30: 30P vs 1.25P + 2.9P = **4.15P → ~86% reduction** on the repeated portion.
- Sensitivity: **one volatile token near the top drops hit rate toward zero** → back to 30P.
- "Agent economics are decided by prompt layout, not model selection … defend the cache boundary as an **architectural
  invariant rather than a tuning parameter**."

**Step 6 — Architecture: order the prompt by volatility.** Caching is prefix-based (longest identical leading token
sequence); a change at position k invalidates everything from k onward.
```
CORRECT: [system prompt][tool schemas][repo map][history steps 1..k][current turn]   -> first 4 segments cached every call
BROKEN:  [timestamp][system prompt][tool schemas][repo map][current turn]             -> nothing cached
```
**Why the system prompt never changes mid-session:** timestamp, turn counter or randomly ordered tool list near the top
invalidates the whole cache every call; hit rate collapses, cost rises ~7× (30P vs 4.15P), **nothing functional breaks — no
test fails, output unchanged** — invisible except on a cost dashboard. "The most common and most expensive self-inflicted
wound in agent design."

Six sub-layers:
- **(a) Prompt assembly** — fixed stable-to-volatile order as a **system invariant**; enforce in code by **hashing the
  assembled prefix and asserting the hash stable across a session**.
- **(b) Cache key construction & block hashing** — matched block-by-block from the start; a mid-block change invalidates
  the whole block.
- **(c) Prefix-aware routing** — cache lives in one replica's memory; round-robin scatters a session; **session affinity**
  is the simplest recovery.
- **(d) Eviction & lifetime policy** — a capacity decision: too short → long tasks lose prefix mid-session; too long →
  memory held for ended sessions.
- **(e) Invalidation on change** — stale mapped file/schema = agent reasoning about a file that no longer exists:
  **correctness** failure, not cost.
- **(f) Accounting & attribution** — per-session and per-deploy hit rate/tokens/spend; without it a cost regression is
  found from the monthly invoice.

**What is physically cached:** KV cache size = **2 × L × H_kv × d_h × S × b** (layers, KV heads, head dim, sequence length,
batch). A **memory-bandwidth artefact, not compute** → serving is a memory-management problem; prefix caching reuses this
store *across* requests, bounded by the same GPU memory.

**Step 7 — Evaluation.**
| Metric | Tells / hides |
|---|---|
| Cache hit rate | Primary health signal; hides usefulness (a cached irrelevant repo map scores 100%) |
| **Tokens per resolved task** | The metric that matters — efficiency + whether task finished |
| Cost per resolved task | Priced; mean hides the tail of runaway sessions |
| TTFT p50/p95 | A miss surfaces as latency before cost — often earliest signal |

**Step 8 — A/B: "the easiest experiment in the document".** Cache changes can be randomised **at request level** because
they **don't alter the output distribution** — same content reaches the model; only billing path and latency differ; t-test
assumptions hold; can't change user behaviour; requests don't interfere. (Contrast: interventions that change the output
break all of these.)

**Step 9 — Deployment & monitoring.** Cache hit rate as **first-class SLO, alert on a drop** (not absolute threshold);
**regression-test hit rate in CI** — hash assembled prefix, fail build on unexpected change ("converting an invisible cost
regression into a test failure"); **attribute cost per deploy**; **alert on the tail** (runaway sessions move p99 before the mean).

**Step 10 — Risks.**
- **Cross-tenant cache leakage** — content-only keys let tenants share blocks containing proprietary source →
  confidentiality failure; **namespace keys by tenant unconditionally**.
- **Serving stale cached context** — confidently wrong patches; every cached segment needs a **validity condition tied to
  the underlying artefact**, not only time-based expiry.
- **Vendor cache semantics are not a stable contract** — lifetimes, price ratios, min cacheable lengths change; measure
  realised saving continuously.
- **Over-fitting the prompt to the cache** — freezing the prefix discourages improving system prompt/repo map; **batch
  prompt changes into versioned releases**, accepting a one-off reset.

### Likely doubts & exam angles (coding agent)
- **Q: Why does cost scale ~N²?** A: Each call resends the growing history; cumulative tokens ≈ sum of a linearly growing series.
- **Q: Why isn't a cheaper model the fix?** A: Lowers unit price but not quadratic growth, and usually increases step count.
- **Q: Break-even for caching?** A: 1.25 + 0.1(n−1) < n ⇒ n ≥ 2; at n = 30, ~86% saving (30P → 4.15P).
- **Q: What silently kills the cache?** A: Any volatile token near the prompt head (timestamp, turn counter, shuffled tools) — no functional failure, only cost.
- **Q: How to catch it?** A: Hash the assembled prefix, assert stable per session, fail CI on unexpected change; hit-rate SLO with drop alerts; per-deploy cost attribution.
- **Q: Why can cache changes be A/B'd at request level?** A: They don't change model output → no behaviour change, no interference.
- **Q: Why does round-robin load balancing hurt?** A: Cache lives on one replica; round-robin scatters the session → misses; use session affinity / cache-aware routing.
- **Q: Security risk of prefix caching?** A: Cross-tenant leakage of proprietary code if keys are content-only → namespace by tenant.
- **Q: Why is accept rate a weak online metric?** A: Biased upward — engineers accept code they haven't read; revert and follow-up-turn rates are more honest.
- **Q: What are PREWARM/LOOP/POST lanes?** A: Once-paid setup, the ×30 hot loop that dominates cost, and async post-processing on cheaper models — hoist/defer work out of the hot lane.

---

## Cross-links
- **Rubric** reused from W7S1 across W8S1 and both bonus case studies; W8 variant adds "cost of a wrong action", "what the
  agent decides at each step", severity bias, and "when NOT to use agents".
- **W8S1 ↔ W7S1:** both use DistilBERT intent routing, Redis session state, circuit breakers, canary for LLM updates,
  PII scrubbing before logging, session as A/B unit. Ride-hailing model 56 explicitly = "the W8S1 architecture, deployed on
  ride disputes".
- **W8S1 ↔ W7S2:** prompt-injection defences, cost-per-request ceilings (W7S2 Incident 3), audit/trace logging (Langfuse),
  LLM-as-judge evaluation.
- **Coding agent ↔ W7S2:** LLM-as-judge with bias controls, cost attribution per deploy, injection via retrieved content,
  "every prompt change is a deployment event" → versioned prompt releases; **↔ W8S1:** loop detection, budget/step
  ceilings, permission gating (human approval), bounded execution.
- **Coding agent KV cache formula, continuous batching, speculative decoding, prefill/decode split** connect to earlier LLM
  inference-serving material.
- **Ride-hailing ↔ coding agent:** "move the model out of the request path / move work out of the hot lane"; "estimation
  selects the architecture"; feedback loops where the system trains on its own decisions (dispatch → demand data;
  trajectories → preference data); permanent randomisation budgets (surge jitter; policy-confounding exploration budget).
- **Ride-hailing A/B (SUTVA, switchback, geo)** extends the A/B testing material from earlier weeks.

## Glossary
- **Agent runner** — orchestration layer that enforces hard limits (max turns, cost ceiling) outside the LLM.
- **Bounded execution** — design principle: cap loops, cost, context and tool waits rather than optimise.
- **Intent router** — fast classifier (DistilBERT, < 20 ms) that routes messages to tool/KB/agent/escalation.
- **Function calling** — LLM emits structured tool calls (e.g. `get_order_status(order_id)`).
- **Loop detector** — same tool + same args twice → break and escalate.
- **Sliding window + summarisation** — keep last N turns verbatim, compress older turns into a memory string.
- **Human approval gate** — high-stakes actions (refund > $50) need human sign-off.
- **Append-only audit log** — immutable record of every tool call, args, result, LLM response, latency.
- **Severity bias** — AI handles easy cases, humans get hard ones → AI metrics look better than they are.
- **Resolution rate** — % conversations resolved without human (resolved = no re-contact in 24h).
- **CSAT** — customer satisfaction score (1–5 stars).
- **SUTVA** — Stable Unit Treatment Value Assumption; one unit's treatment doesn't affect another's outcome.
- **Switchback test** — alternate a whole region between policies over randomised time periods; cluster SEs by period.
- **Geo/cluster randomisation** — assign whole cities/clusters; analyse with DiD or synthetic control; buffer zones.
- **Carryover** — residual effect of the previous policy period in a switchback.
- **Censored demand** — completed trips = min(demand, supply); true demand unobserved where supply binds.
- **Tobit model** — censored-likelihood regression for lower-bounded observations.
- **H3 / k-ring** — hexagonal geo grid (Uber) / neighbouring cells within k steps.
- **Seasonal-naive** — baseline forecast: same cell, same hour, last week.
- **WAPE** — weighted absolute percentage error; robust to zero actuals.
- **Pinball (quantile) loss** — loss for quantile predictions.
- **Hierarchical reconciliation (MinT)** — make cell forecasts sum to zone/city forecasts.
- **Point-in-time correctness** — features joined as they were known at prediction time (forecast, not observed weather).
- **Guidance regret** — drivers following forecast guidance earning less than those ignoring it.
- **Upfront fare / risk margin ρ(σ)** — fixed pre-trip quote; buffer scaled by predicted uncertainty.
- **Aggregate quote bias** — mean(quoted − actual); the finance-grade metric for fares.
- **Shadow-quote** — run new fare model silently against metered actuals before exposure.
- **Adverse selection (fare)** — vague-destination, high-variance trips systematically underpriced.
- **Surge multiplier m** — per-cell, per-minute price multiplier in [1.0, m_max]; precomputed lookup.
- **Price elasticity (causal)** — demand/supply response to price, estimated via randomisation, not logs.
- **Randomisation budget** — permanent 1–2% of cell-minutes with jittered prices to keep elasticity honest.
- **Regression discontinuity** — compare units just above/below a threshold.
- **Backward-bending supply** — higher pay → earnings-target drivers work fewer hours.
- **Fail-open / fail-closed** — on timeout allow (fraud screen) vs deny.
- **Uplift modelling** — target units whose behaviour changes because of treatment (persuadables).
- **Prefix caching** — provider reuses KV state for identical leading tokens across requests (write ~1.25×, read ~0.1×).
- **Cache hit rate** — fraction of input tokens served from cache; target > 90% from step 2.
- **Order by volatility** — prompt layout most-stable-first so the cache boundary sits far right.
- **TTFT** — time to first token; cache misses show up here first.
- **Session affinity / cache-aware routing** — route a session to the replica holding its prefix.
- **KV cache** — per-token key/value tensors; size 2·L·H_kv·d_h·S·b; memory-bandwidth bound.
- **Compaction** — summarising/pruning history to bound context growth.
- **Step ceiling** — max steps per task, bounding the N² cost tail.
- **Cost per resolved task** — the coding agent's operative business metric.
- **RLVR** — reinforcement learning with verifiable rewards (tests passing).
- **Accept rate vs revert rate** — upward-biased online proxy vs lagging, more honest signal.
- **PREWARM / LOOP / POST** — once-paid, per-iteration, and async lanes of an agent task.

---

## Course Closing — "From Notebooks to Production Systems"
Source: course_closing_slides.pdf (Priyansh Saxena, ML System Design and LLMOps, Course Closing, September 2026)

| Where we started | Where we are now |
|---|---|
| A model that works on your laptop | Feature stores, serving layers, MLOps pipelines |
| "It gets 94% accuracy, ship it" | LLM inference at scale, RAG, and agentic systems |
| No feature store, no registry, no monitoring | A **10-step framework to design any ML system — and defend it** |

- Closing quote: *"You didn't just learn ML system design — you built the judgment to reason about systems that don't
  exist yet."*
- Thank-you slide: "It's been a privilege watching you turn theory into systems thinking. Now go build the systems nobody's
  written the playbook for yet." — Priyansh Saxena, www.priyanshsaxena.com (feedback QR code).
- (note: course materials also referenced at priyanshsaxena.com/sst-mlsystemdesign in the W7S2 handout.)
