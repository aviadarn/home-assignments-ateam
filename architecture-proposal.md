# AI Architect Technical Exercise — Architecture Proposal

**Prepared for:** A.Team / CPG Client
**Date:** 2026-03-11

---

## Table of Contents

1. [Data Flow Diagram](#1-data-flow-diagram)
2. [Component Breakdown](#2-component-breakdown)
3. [Core Technical Approach](#3-core-technical-approach)
4. [Learning Loops & System Improvement](#4-learning-loops--system-improvement)
5. [Learning Loop Code Sample](#5-learning-loop-code-sample)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Data Model & Storage](#7-data-model--storage)
8. [Tradeoffs & Risks](#8-tradeoffs--risks)

---

## 1. Data Flow Diagram

See [`data-flow-diagram.md`](./data-flow-diagram.md) for the full Mermaid diagram with component annotations and schemas.

**Summary:** Social posts travel through a 4-stage funnel — ingestion, processing, alerting, and feedback — with three parallel learning loops that continuously update query weights, relevance thresholds, and the LLM prompt. The lifecycle of a single post: API retrieval → embedding → dedup → velocity scoring → brand similarity filter → relevance classifier → LLM scorer → alert → dashboard → feedback → learning engine → updated parameters.

---

## 2. Component Breakdown

### 2.1 Brand Profile Extractor

**Purpose:** Transform unstructured brand guideline PDFs into structured, queryable brand profiles.

**Inputs:** PDF bytes from brand guidelines repository
**Outputs:** `brand_profile` JSONB record (themes, keywords, off-limits list, audience, profile embedding centroid)

**How it works:** An LLM (Claude claude-sonnet-4-6 or GPT-4o) receives the full PDF text and is prompted to extract: positive themes (topics to engage), exact keywords for API queries, off-limits topics as a separate list, competitor brand names, and audience psychographics. The extracted keywords are embedded using all-MiniLM-L6-v2 and averaged into a 384-dim profile centroid stored alongside the structured data.

**Failure modes:** PDF parsing failure (handled by text extraction fallback); LLM extraction hallucination (mitigated by structured output schemas + human review on initial extraction); stale profiles when PDFs are updated (mitigated by version tracking + re-extraction webhook).

**Technologies:** PyMuPDF for PDF extraction, Anthropic Claude API for structured extraction, sentence-transformers for embedding, Postgres for storage.

**Key design decision:** Storing the profile centroid embedding alongside the structured data allows the Candidate Filter to do fast ANN search without re-calling the LLM on every post. The LLM is only called during profile extraction (infrequent) and final alert generation (selective).

---

### 2.2 Query Generator

**Purpose:** Translate brand profiles + learned term weights into boolean API queries, scheduled per-brand.

**Inputs:** `brand_id`, current `query_terms[]` with weights
**Outputs:** POST `/v1/search` request bodies; log entries in `query_log`

**How it works:** For each brand, fetches all `query_terms` where weight > 0.3. Constructs a boolean query: core terms joined with OR, expansion terms with weight > 0.6 added as OR extensions, negative terms prepended with NOT. Queries are issued every 30 minutes per brand on a rotating schedule (to avoid thundering herd across 40+ brands, uses a hash-distributed cron offset).

**Example output for Zephyr Energy:**
```
("energy drink" OR focus OR productivity OR "deep work" OR "afternoon slump" OR "clean energy")
AND NOT (weightloss OR "extreme sports" OR "all nighter" OR sponsored OR #ad)
```

**Failure modes:** API rate limiting (backoff with jitter); query syntax errors (validated pre-send against known operator list); stale term weights (Learning Engine resets weights to guidelines baseline if no feedback received in 14 days).

**Technologies:** Python scheduler (APScheduler), Postgres for term weights, requests library.

---

### 2.3 Post Ingester

**Purpose:** Fetch, paginate, and durably store all posts returned by the Social Listening API.

**Inputs:** API responses (JSON)
**Outputs:** Raw post JSONL files in object store, `post_metadata` records in Postgres

**How it works:** Paginates the API using the `cursor` field until all results for a query window are fetched. Deduplicates by `post.id` before writing — if a post appears in multiple brand queries, it is stored once in object store with a `brand_associations[]` field. Writes a lightweight metadata record to Postgres (post_id, brand_ids, created_at, platform, retrieved_at) for efficient downstream querying.

**Failure modes:** API downtime (queued retry with 5-minute backoff, up to 3 retries); partial pagination failure (tracks cursor in Postgres, resumes from last successful page); duplicate posts (idempotent insert on post_id).

**Technologies:** S3 for raw storage, Postgres for metadata, Python worker (Celery or AWS Lambda).

**Storage rationale:** Object store for raw posts because post text and engagement data are schema-flexible across platforms (Reddit has subreddit, TikTok has video_duration, etc.), volumes are high, and raw data should be immutable for reprocessing.

---

### 2.4 Processing Pipeline

Four sequential steps running as a streaming or micro-batch job (every 15 minutes):

**Step 1 — Embedder:** Batch-encodes post text using all-MiniLM-L6-v2 (384-dim, ~2,000 posts/second on CPU). Writes embeddings to pgvector indexed table.

**Step 2 — Near-Duplicate Filter:** For each new post embedding, queries pgvector for any stored post with cosine similarity > 0.97. Drops the new post if a near-duplicate exists (retains higher-engagement copy). Reduces noise from reshares and paraphrased reposts.

**Step 3 — Velocity Scorer:** Calls `/v1/metrics/velocity` for each active query topic. Tags posts belonging to topics with `engagement_growth_rate > 1.5` over the past 6 hours as "trending." Posts from flat topics are deprioritized.

**Step 4 — Candidate Filter:** Computes cosine similarity between each post embedding and the brand profile centroid. Posts below 0.55 are dropped (~90% of posts, preserving API cost efficiency). This is not a hard relevance gate — it's a cheap pre-filter before the expensive steps.

**Step 5 — Relevance Classifier:** Runs a lightweight logistic regression model on a 12-feature vector:
- cosine_similarity_to_brand_centroid (float)
- velocity_score (float)
- log(total_engagement + 1) (float)
- platform_match_score (float, 1.0 if on brand's primary platforms)
- off_limits_keyword_hit (bool, 0/1)
- author_follower_count_log (float)
- post_age_hours (float)
- brand_competitor_mention (bool)
- has_hashtag (bool)
- subreddit_relevance_score (float, 0 if not Reddit)
- is_news_source (bool)
- weekend_flag (bool)

Output: `relevance_score` float in [0, 1]. Only posts above a per-brand threshold (learned, default 0.65) proceed to LLM scoring.

**Failure modes:** Embedding service timeout (retry + fallback to TF-IDF similarity); classifier model load failure (fallback to cosine similarity only with raised threshold 0.70).

---

### 2.5 LLM Scorer & Explainer

**Purpose:** Final relevance gate + generate human-readable alert explanation.

**Inputs:** Post text, brand profile (themes, off-limits, voice), trend cluster context
**Outputs:** `{relevance: float, explanation: str, off_limits: bool, why_relevant: str}`

**Prompt structure (simplified):**
```
You are evaluating whether a social media trend is relevant to a brand.

Brand: {brand_name}
Brand themes: {themes}
Off-limits topics: {off_limits}
Brand voice: {voice_description}

Trend cluster (top 3 posts by engagement):
{post_texts}

Task: Score relevance 0.0–1.0. If off-limits topics appear, set off_limits=true regardless of score.
Explain in 1-2 sentences why this trend is (or isn't) relevant to this brand.

Respond as JSON: {"relevance": float, "off_limits": bool, "why_relevant": str}
```

**Failure modes:** LLM latency spikes (async with 10-second timeout; fallback to classifier score only, no explanation); hallucinated off-limits flags (threshold at 0.85 for off-limits suppression, not 1.0, allowing human review).

**Cost control:** Only top-20 posts per brand per 30-minute cycle reach the LLM (filtered by classifier). At 40 brands × 20 posts × 48 cycles/day = 38,400 LLM calls/day. At ~$0.003/call, approximately $115/day — well within managed service budget.

---

### 2.6 Alert Assembler

**Purpose:** Group scored posts into coherent trend alerts; rank for brand manager consumption.

**Inputs:** LLM-scored posts (per brand, per cycle)
**Outputs:** `Alert` records in Postgres

**How it works:** Applies HDBSCAN clustering on post embeddings (min_cluster_size=3) to group posts discussing the same trend. For each cluster: selects the highest-engagement post as the representative, uses the LLM's `why_relevant` explanation, and computes a composite score:

```
composite_score = 0.35 × relevance_score
               + 0.35 × velocity_score
               + 0.20 × log(cluster_engagement + 1) / log(max_engagement + 1)
               + 0.10 × novelty_score  # 1 if not seen in prior 48h, decaying
```

Alerts with `off_limits=true` are suppressed entirely. Alerts above `composite_score > 0.60` are published to the dashboard. Max 10 alerts per brand per day to prevent alert fatigue.

---

### 2.7 Dashboard

**Purpose:** Deliver prioritized alerts to brand managers; collect behavioral feedback.

**Inputs:** `Alert` records from Postgres
**Outputs:** Rendered alert cards; `alert_feedback` events

**Design:** REST API backend (FastAPI) serving a React frontend. Each alert card shows: trend headline, why-it's-relevant explanation, 2-3 representative posts, engagement velocity chart, and action buttons (Share to Team, Create Content Brief, Dismiss). All interactions are logged with timestamp and dwell time.

---

### 2.8 Feedback Collector & Learning Engine

See Section 4 (Learning Loops) for full specification.

---

## 3. Core Technical Approach

### 3.1 Query Construction

The Social Listening API requires explicit search parameters — we cannot observe trends we don't query for. This is the system's most critical design constraint.

**Phase 1 — Seed query construction:** The Brand Profile Extractor uses an LLM to extract two query layers from each brand guideline PDF:
- **Core terms** (always included): brand-defining topics like `"energy drink"`, `"focus"`, `"morning routine"` for Zephyr
- **Contextual terms** (rotated): audience behaviors, competitor names, seasonal themes

**Phase 2 — Query expansion via learning loop:** Over time, the Query Expansion Learning Loop (Section 4.1) discovers new terms from posts that brand managers engaged with — terms not in the original guidelines. These become weighted expansion terms appended to base queries.

**Phase 3 — Negative filtering:** Off-limits topics from brand guidelines are converted to NOT clauses (e.g., `NOT ("extreme sports" OR "weight loss" OR "all nighter")`). This pre-filters at the API level before any expensive processing.

**Handling 40+ brands:** Queries are staggered on a 30-minute schedule using brand_id hash % 60 as a minute offset. Each brand gets approximately 48 query cycles per day. Brands share no query state — each has its own term weight table.

---

### 3.2 Representation

| Entity | Representation | Rationale |
|---|---|---|
| Brand profile | Structured JSON + 384-dim embedding centroid | Centroid enables fast ANN filtering; structured fields drive query construction and LLM prompts |
| Post text | 384-dim dense embedding (all-MiniLM-L6-v2) | Strong semantic similarity without fine-tuning; fast CPU inference; 384-dim balances quality and storage |
| Trends/clusters | HDBSCAN clusters over post embeddings + representative post | No fixed cluster count; handles noise posts as unclustered; clusters are natural groupings of conversation |
| Learned query weights | Float weights per (brand_id, term) in Postgres | Queryable, versionable, auditable; allows per-brand customization |
| Off-limits topics | Keyword list + optional embedding-based classifier | Keywords for fast blocking; embedding classifier catches paraphrases not caught by exact keywords |

**Why not fine-tuned embeddings?** The 2-3 engineer team constraint argues against maintaining a fine-tuned embedding model. all-MiniLM-L6-v2 is general-purpose and strong enough for semantic filtering. The classifier layer handles brand-specific relevance learning.

**Why not topic modeling (LDA/BERTopic) as the primary representation?** Topic models are trained on a corpus and don't naturally adapt to individual brand dimensions. HDBSCAN over pre-computed embeddings is simpler, doesn't require retraining, and clusters can be inspected directly.

---

### 3.3 Relevance Determination

Three-stage funnel (each stage more expensive but more accurate):

1. **Cosine similarity gate** (threshold 0.55): Fast ANN lookup in pgvector. Eliminates ~90% of posts in milliseconds. Threshold is intentionally permissive — false negatives at this stage are unrecoverable.

2. **Logistic regression classifier** (threshold learned per brand): 12 features combining semantic, engagement, and platform signals. Trained on feedback events. Threshold defaults to 0.65, updated weekly by the Threshold Calibration Loop.

3. **LLM scorer** (top 20 posts per brand per cycle): Final semantic gate with full brand profile context. Also handles off-limits detection and explanation generation. This is the only stage that can reason about nuanced guideline constraints like "avoid 'clean living' sanctimony."

**Why three stages?** The vast majority of posts (>99%) are irrelevant to any given brand. Running LLM scoring on all posts would be cost-prohibitive and slow. The cheap stages are intentionally over-inclusive; the expensive LLM stage is the precision gate.

---

### 3.4 Timeliness

Target: actionable insights within **2–4 hours** of trend emergence.

- **API polling:** Every 30 minutes per brand (staggered)
- **Processing pipeline:** Runs every 15 minutes as a micro-batch
- **Alert assembly:** Triggered after each pipeline run
- **Dashboard push:** WebSocket notifications to logged-in brand managers

**Latency budget for a single post:** API retrieval (0-30 min depending on cycle) + embedding (< 1 sec) + processing pipeline (< 5 min for full batch) + LLM scoring (< 10 sec per post, async) + alert assembly (< 1 min) = **worst case ~37 minutes from post creation to alert delivery**, well within the hours target.

**Why not streaming (Kafka + Flink)?** The API is query-based, not a firehose. True streaming would require continuous polling anyway. Micro-batch every 15 minutes achieves comparable latency without the operational overhead of a streaming stack — appropriate for a 2-3 engineer team.

---

### 3.5 Prioritization

Brand managers cannot review everything. Posts are ranked by composite score (defined in Section 2.6). Additionally:

- **Brand-level cap:** Maximum 10 alerts per brand per day (prevents fatigue)
- **Novelty decay:** Alerts about the same trend cluster from the prior 48 hours get a `novelty_score` penalty
- **Urgency boost:** Trends with `engagement_growth_rate > 3.0` (viral velocity) skip the daily cap and generate an immediate push notification
- **Personalization (future state):** Once sufficient per-manager feedback accumulates, composite weights can be personalized per manager

---

### 3.6 Quality Controls

| Problem | Control Mechanism |
|---|---|
| **Noise / low-quality posts** | Minimum engagement threshold (likes + reposts > 5); bot detection via follower/engagement ratio check |
| **Near-duplicate posts** | Cosine similarity dedup in vector store (threshold 0.97) |
| **Off-limits content** | NOT clauses in API query + keyword list + LLM off-limits flag; triple-layered |
| **Topic drift** | Query term weight decay (×0.9 per week without positive signal) reverts toward guidelines baseline |
| **Adversarial / spam content** | Author account age filter (< 30 days excluded); engagement velocity sanity check (sudden 100× spike flagged for review) |
| **Alert fatigue** | Hard cap of 10 alerts/brand/day + composite score threshold |

---

## 4. Learning Loops & System Improvement

This section describes three learning loops in the system. Each is specified with source data schema, learning target, optimization method, and evaluation criteria.

---

### 4.1 Query Expansion Learning Loop

**What it improves:** The set of search terms used to query the Social Listening API for each brand. Over time, the system discovers new vocabulary that brand managers find valuable — vocabulary not present in the original brand guidelines.

#### A. Source Data Schema

**Table: `query_log`**
```sql
query_log(
  query_log_id    UUID PRIMARY KEY,
  brand_id        TEXT NOT NULL,
  query_string    TEXT NOT NULL,           -- the full boolean query sent
  terms_used      TEXT[],                  -- individual terms in query
  post_ids        TEXT[],                  -- all post IDs returned
  surfaced_ids    TEXT[],                  -- post IDs that reached the dashboard
  issued_at       TIMESTAMPTZ NOT NULL,
  window_from     TIMESTAMPTZ,
  window_to       TIMESTAMPTZ
)
```

**Table: `alert_feedback`** (see Section 2.7 for full schema)

**Example records:**
```json
// query_log record
{
  "query_log_id": "a1b2c3d4",
  "brand_id": "zephyr_energy",
  "query_string": "\"energy drink\" AND (focus OR productivity) AND NOT \"extreme sports\"",
  "terms_used": ["energy drink", "focus", "productivity"],
  "post_ids": ["tw_001", "rd_002", "tt_003", "ig_004"],
  "surfaced_ids": ["tw_001", "rd_002"],
  "issued_at": "2025-01-15T14:00:00Z"
}

// alert_feedback record (brand manager clicked)
{
  "feedback_id": "f9e8d7c6",
  "alert_id": "alert_abc",
  "brand_id": "zephyr_energy",
  "brand_manager_id": "mgr_sarah",
  "action": "content_brief",
  "dwell_time_ms": 42000,
  "created_at": "2025-01-15T14:35:00Z"
}
```

**Volume estimate:** 40 brands × 48 queries/day × 7 days = ~13,440 query_log records/week. ~200 feedback events/week across all brands at launch (scales with active users).

**Collection:** query_log written by Query Generator on every API call. alert_feedback written by dashboard on every user interaction.

---

#### B. Learning Target

**Target:** `query_terms` table — specifically the `weight` column for expansion terms per brand.

**Current form:**
```
query_terms(brand_id, term, weight, term_type, source, updated_at)
```
Terms with `term_type = 'expansion'` and `source = 'learning_loop'` are the learnable parameters. Each is a float in [0.0, 1.0]. Terms below 0.3 are pruned from queries; terms above 0.6 are included as OR extensions to the core query.

**Example current state for Zephyr:**
```
core:      "energy drink" (1.0), "focus" (1.0), "productivity" (1.0)
expansion: "afternoon slump" (0.82), "deep work" (0.71), "L-theanine" (0.65)
           "nootropics" (0.41), "work from home" (0.38)
```

---

#### C. Optimization Method

**Weekly batch job** (runs every Monday 02:00 UTC):

**Step 1 — Collect positive examples:** Join `query_log` + `alert_feedback` to find posts surfaced in the past 7 days that received `action IN ('click', 'content_brief', 'share')`.

**Step 2 — Extract n-gram candidates:** From positive example post texts, extract all 1-3 word n-grams. Remove stopwords, existing core terms, and n-grams appearing in off-limits list.

**Step 3 — Score candidates:**
```python
score = click_rate * action_rate * 10 + math.log(frequency + 1)
# click_rate = posts_with_ngram_clicked / posts_with_ngram_surfaced
# action_rate = posts_with_ngram_actioned / posts_with_ngram_clicked
# frequency = count of posts containing this n-gram in the week
```

**Step 4 — Promote new terms:** N-grams with `score > 0.5` AND `frequency >= 3` are inserted as new expansion terms with initial weight = `min(score / 5.0, 0.95)`.

**Step 5 — Update existing weights:** For existing expansion terms, update weight using:
```python
new_weight = 0.9 * old_weight + 0.1 * current_score
# Exponential moving average: recent signal has 10% influence per cycle
# If no posts containing term were surfaced this week, score = 0, weight decays by 0.9
```

**Step 6 — Prune dead terms:** Remove expansion terms where `weight < 0.3` for 3 consecutive weeks.

**Step 7 — Write versioned snapshot:** Insert new `query_terms` records with `updated_at = NOW()`. Old weights remain queryable for audit/rollback.

---

#### D. Evaluation Criteria

**Primary metric:** Click-through rate (CTR) on surfaced alerts.
```
CTR = count(actions IN {click, content_brief, share}) / count(alerts delivered)
```

**Baseline:** 12% CTR (estimated from current weekly manual process; brand managers click ~1 in 8 alerts as meaningful)

**Target:** 18% CTR within 8 weeks of learning loop activation

**Secondary metrics:**
- Query coverage: `unique terms in queries` — should grow over first 4 weeks then plateau
- Term churn rate: terms added / terms dropped per cycle — high churn indicates unstable learning
- Alert volume: total alerts/day — should remain ≤ 400 across all brands (10/brand × 40 brands)

**Guardrail:** If CTR drops below 8% for any brand in a given week, that brand's expansion terms are reset to the guidelines baseline and a human review is triggered.

---

### 4.2 Relevance Threshold Calibration Loop

**What it improves:** The per-brand relevance score threshold applied to the logistic regression classifier (Step 5 of the Processing Pipeline). The threshold determines how strictly the classifier filters before sending to the LLM.

#### A. Source Data Schema

**Table: `alert_feedback`** (same as above)

**Table: `alerts`** (see Section 2.7)

Key fields used: `alert_id`, `brand_id`, `relevance_score` (classifier output), `action` from feedback.

**Example records:**
```json
// High-scoring alert, brand manager dismissed it
{"alert_id": "a001", "brand_id": "brighten_home", "relevance_score": 0.78, "action": "dismiss", "dwell_time_ms": 800}

// Lower-scoring alert, brand manager created a content brief
{"alert_id": "a002", "brand_id": "brighten_home", "relevance_score": 0.67, "action": "content_brief", "dwell_time_ms": 65000}
```

**Volume estimate:** ~400 alerts/day × 30% feedback rate = ~120 labeled examples/day; ~840/week across all brands.

---

#### B. Learning Target

**Target:** `brand_relevance_threshold` — a per-brand float (default 0.65) stored in `brand_profiles.classifier_threshold`.

**Current form:** A single threshold applied to the classifier score before LLM scoring. Currently global (0.65 for all brands). The loop individualizes this per brand based on observed precision/recall tradeoffs.

---

#### C. Optimization Method

**Bi-weekly batch job:**

**Step 1:** For each brand with ≥ 50 feedback events in the past 2 weeks, collect `(relevance_score, label)` pairs where label = 1 if action ∈ {click, content_brief, share}, else 0.

**Step 2:** Sweep threshold values from 0.40 to 0.90 in steps of 0.05. For each threshold, compute:
```python
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
f_beta    = (1 + beta²) * (precision * recall) / (beta² * precision + recall)
# beta = 0.5: weight precision 2× higher than recall
# (brand managers prefer fewer, higher-quality alerts)
```

**Step 3:** Select threshold maximizing F_0.5 score. Apply a ±0.05 dampening to avoid large jumps: `new_threshold = old_threshold + clip(optimal - old_threshold, -0.05, +0.05)`.

**Step 4:** Write new threshold to `brand_profiles.classifier_threshold` with version increment.

**Guardrail:** Threshold is bounded to [0.45, 0.85]. If optimal threshold would be outside this range, flag for human review rather than applying.

---

#### D. Evaluation Criteria

**Primary metric:** Precision@10 — of the 10 daily alerts surfaced to a brand manager, how many receive a positive action?
```
Precision@10 = positive_actions / 10
```

**Baseline:** 2.5/10 (25% — estimated from analogous B2B content tools)

**Target:** 4/10 (40%) within 12 weeks

**Secondary:** Recall proxy — track whether managers report (via dashboard feedback button) that they "missed something important." This is an imperfect recall signal but the only one available without ground truth labeling.

---

### 4.3 Prompt Optimization Loop

**What it improves:** The LLM prompt template used to score relevance and generate explanations. Poor prompts cause systematic errors — for example, the LLM consistently marking sustainability content as off-limits for Brighten Home when the brand actually wants to engage with sustainability (just not "political environmental activism").

#### A. Source Data Schema

**Table: `alert_prompt_log`**
```sql
alert_prompt_log(
  prompt_log_id   UUID PRIMARY KEY,
  alert_id        UUID REFERENCES alerts(alert_id),
  brand_id        TEXT,
  prompt_version  TEXT,          -- e.g., 'v3.2'
  prompt_text     TEXT,          -- full prompt sent to LLM
  llm_response    JSONB,         -- {relevance, off_limits, why_relevant}
  latency_ms      INTEGER,
  created_at      TIMESTAMPTZ
)
```

**Example records:**
```json
// Alert suppressed by LLM off-limits flag, but brand manager later marked similar content as valuable
{
  "prompt_log_id": "pl_001",
  "alert_id": "a003",
  "brand_id": "brighten_home",
  "prompt_version": "v3.1",
  "llm_response": {
    "relevance": 0.82,
    "off_limits": true,
    "why_relevant": "Post discusses political environmental activism which is off-limits"
  }
}
```

**Volume estimate:** ~400 prompt_log records/day. Negative feedback events (dismissals of shown alerts + suppressed alerts that were later discovered) = ~50-100/week per brand.

---

#### B. Learning Target

**Target:** The LLM prompt template — specifically the instructions section describing how to interpret off-limits constraints and relevance scoring criteria.

**Current form:** A versioned string stored in S3 at `s3://prompt-store/{brand_id}/relevance_prompt_v{N}.txt`. The active version pointer is stored in `brand_profiles.prompt_version`.

---

#### C. Optimization Method

**Monthly prompt review cycle (semi-automated, human-in-the-loop):**

**Step 1 — Failure analysis:** Collect past month's alerts where: (a) LLM flagged as off-limits AND brand manager later marked similar content as valuable, OR (b) alert was delivered but immediately dismissed with dwell_time < 2 seconds (strong negative signal). These are "failure cases."

**Step 2 — Pattern extraction (LLM-assisted):**
```
Prompt to analysis LLM:
"Here are {N} cases where our relevance scoring prompt produced poor results for brand {brand_name}.
For each case, the post text and brand guidelines are provided.
Cases: {failure_case_examples}

Identify the 2-3 most common failure patterns. For each pattern:
1. Describe what the prompt misunderstood
2. Suggest a specific rewording of the relevant prompt section

Format: JSON array of {pattern, current_text, suggested_revision}"
```

**Step 3 — Candidate prompt construction:** Apply suggested revisions to current prompt template. Increment version.

**Step 4 — A/B evaluation:** Route 20% of alerts for each affected brand to the new prompt version. Compare CTR and off-limits suppression rate over 1 week.

**Step 5 — Rollout or rollback:** If new prompt version shows CTR improvement > 5% with no increase in off-limits suppression rate, promote to 100%. Otherwise, rollback and flag for human review.

**Key constraint:** Prompt changes are never applied without A/B evaluation. The old version remains available in S3 for rollback. All alerts carry a `prompt_version` field enabling clean experiment tracking.

---

#### D. Evaluation Criteria

**Primary metric:** Off-limits suppression accuracy
```
suppression_precision = true_off_limits_suppressions / total_suppressions
# measured by sampling 50 suppressed alerts/month for human review
```

**Baseline:** Estimated 75% suppression precision (25% false positives where content was actually relevant)

**Target:** 90% suppression precision within 6 months

**Secondary:** CTR delta between old and new prompt version during A/B period (target: ≥ +5% relative improvement).

---

## 5. Learning Loop Code Sample

See [`../infra/src/learning_loop.py`](../infra/src/learning_loop.py) for the complete working implementation of the Query Expansion Learning Loop.

The code:
1. Generates a mock feedback dataset (50 alerts, 3 brands, realistic click/action signals)
2. Runs the query expansion optimization procedure
3. Calculates CTR before and after learning
4. Demonstrates overfitting on the mock dataset (CTR improvement on training data)
5. Outputs updated term weights per brand

---

## 6. Evaluation Framework

### Success Metrics (KPIs)

| Metric | Formula | Target | Measurement |
|---|---|---|---|
| Alert CTR | positive_actions / alerts_delivered | ≥ 18% | Daily, per brand |
| Time-to-action | median time from trend_created_at to first brand manager action | ≤ 4 hours | Weekly aggregate |
| False positive rate | dismissed_alerts / total_alerts | ≤ 35% | Weekly, per brand |
| Coverage recall (proxy) | "missed trend" reports / total trends in period | ≤ 10% | Monthly survey |
| Off-limits suppression precision | true_off_limits / total_suppressed (sampled) | ≥ 90% | Monthly human audit |
| System latency P95 | time from post created_at to alert on dashboard | ≤ 2 hours | Continuous |

### Offline Evaluation

**Component-level:**
- **Brand Profile Extractor:** Human eval on 10 brands — do extracted themes, keywords, and off-limits match the source PDF? Target: 95% keyword coverage, 0% off-limits hallucination.
- **Relevance Classifier:** Labeled evaluation set of 500 (post, brand) pairs — manually labeled as relevant/irrelevant by a brand manager. Measure AUC-ROC. Target: AUC > 0.82.
- **LLM Scorer:** Same 500-pair eval set. Measure agreement with human labels. Target: 88% agreement, < 5% false off-limits flags.

**End-to-end:**
- Replay last 2 weeks of social data through the pipeline. Compare surfaced alerts against a "gold standard" set of trends identified post-hoc by brand managers during those weeks. Measure recall@K.

### Online Evaluation

- **A/B testing infrastructure:** New model versions (classifier, prompt) are rolled out to 20% of traffic before full deployment. Statistical significance required before promotion (p < 0.05, minimum 200 alert impressions per variant).
- **Real-time dashboard:** System health metrics (pipeline latency, alerts generated/hour, LLM error rate) monitored via Grafana. Alerting if P95 latency > 4 hours or CTR drops below 8% for any brand.

### Baseline Comparison

| Approach | Alert Frequency | Brand Manager Effort | Estimated CTR |
|---|---|---|---|
| Current (weekly manual reports) | Weekly | High (manual analysis) | ~12% (extrapolated) |
| Simple keyword monitoring | Hourly | Medium (noisy feed) | ~8% (estimated) |
| This system (launch) | Continuous, ≤ 4h | Low | Target: 18% |
| This system (after 8 weeks learning) | Continuous, ≤ 4h | Low | Target: 25% |

---

## 7. Data Model & Storage

### Key Entities & Relationships

```
Brand (1) ──────────── (N) BrandProfile (versioned)
Brand (1) ──────────── (N) QueryTerms (weighted, versioned)
Brand (1) ──────────── (N) QueryLog
QueryLog (1) ────────── (N) Post (via returned_post_ids)
Post (1) ────────────── (1) PostEmbedding
Post (N) ────────────── (N) Brand (many-to-many via brand_associations)
Post (N) ────────────── (1) TrendCluster
TrendCluster (1) ───── (N) Alert
Alert (1) ────────────── (N) AlertFeedback
Alert (1) ────────────── (1) PromptLog
Brand (1) ──────────── (N) ModelVersion (classifier, prompt, thresholds)
```

### Storage Choices

| Store | Technology | What Lives Here | Rationale |
|---|---|---|---|
| **Raw posts** | S3 (object store) | JSONL files partitioned by `brand_id/date/` | Append-only, schema-flexible across platforms, cheap at scale, enables reprocessing |
| **Post embeddings** | pgvector (Postgres extension) | `(post_id, embedding[384], brand_id)` | Enables ANN search with SQL joins; avoids separate vector DB operational overhead for a small team |
| **Alerts, feedback, query logs** | Postgres | All structured operational tables | ACID transactions; complex joins between alerts ↔ feedback ↔ query_log; familiar to any engineer |
| **Brand profiles & query weights** | Postgres (JSONB + structured) | `brand_profiles`, `query_terms` tables | Structured for efficient reads; JSONB for flexible profile fields; version columns for rollback |
| **Model artifacts** | S3 + versioned paths | Classifier `.pkl` files, prompt templates `.txt` | Versioned by convention (`v{N}`); cheap, durable, accessible; active version pointer in Postgres |
| **Session/cache** | Redis | Velocity scores, active session state, pipeline cursors | Low-latency reads; acceptable if lost (recomputable); TTL-based expiry |
| **Feature store** | Not needed at launch | — | With only 12 features computed on-the-fly from Postgres, a dedicated feature store adds operational overhead without proportional benefit |

### Query Patterns & Latency

| Pattern | Frequency | Latency Requirement | Solution |
|---|---|---|---|
| Fetch active query terms for brand | Every 30 min | < 50ms | Postgres index on (brand_id, weight DESC) |
| ANN search for candidate filter | Every post | < 100ms | pgvector HNSW index on embedding column |
| Fetch alerts for dashboard | Per page load | < 200ms | Postgres index on (brand_id, composite_score DESC, created_at DESC); paginated |
| Feedback event insert | Per interaction | < 100ms | Postgres with async write; index on (alert_id, brand_id) |
| Learning loop batch reads | Weekly | No hard requirement | Full table scans acceptable; run off-peak |

### Feature/Parameter Storage

Learned parameters are stored as versioned records in Postgres (query term weights, classifier thresholds) with an `updated_at` timestamp and a `version` integer. Model artifacts (serialized classifiers, prompt files) are stored in S3 with the naming convention `s3://ml-artifacts/{brand_id}/{component}/{version}/`. The active version for each component is a pointer in the `brand_profiles` table (`classifier_version`, `prompt_version` columns). Rollback is performed by updating the pointer — old artifacts are never deleted.

---

## 8. Tradeoffs & Risks

### Tradeoff 1: Query-Based API = Blind Spots

**What we're optimizing for:** Cost efficiency and simplicity. A firehose would be more comprehensive but requires a different (and more expensive) data contract.

**What we're sacrificing:** We can only surface trends we know to query for. A truly novel trend — one using vocabulary not yet in our term lists — may go undetected until the Query Expansion Loop catches up.

**Mitigation:** Run weekly "exploration queries" using broad audience-level terms (e.g., `"young professionals" AND (trend OR viral OR everyone)`) specifically designed to surface unexpected vocabulary. Feed discoveries into the expansion loop as candidates.

**Residual risk:** Low-to-medium. The expansion loop typically surfaces new vocabulary within 1-2 weeks of it appearing in brand-adjacent content.

---

### Tradeoff 2: Embedding Similarity Threshold = Recall vs. Precision

**What we're optimizing for:** Cost and latency. Setting the similarity threshold at 0.55 drops ~90% of posts before expensive processing.

**What we're sacrificing:** Posts that are semantically distant from the brand centroid but still contextually relevant (e.g., a meme format that becomes culturally relevant to Zephyr's audience but uses no productivity vocabulary).

**Mitigation:** The threshold is intentionally permissive (0.55 is low for cosine similarity). Monitor false negative rate via the "missed trend" survey. If recall degrades, lower threshold or add a separate "cultural moment" detection pipeline using engagement velocity alone (no brand similarity gate).

**Residual risk:** Low. Meme-format trends are edge cases; core business value is in topic-aligned trends.

---

### Tradeoff 3: Small Feedback Dataset = Slow Learning

**What we're optimizing for:** Getting to market quickly with a functional system. The classifier and learning loops are included in V1.

**What we're sacrificing:** The classifier and learning loops require labeled data to be effective. With 40 brands and potentially only a handful of active brand managers per brand, feedback volume may be low at launch (< 50 events/brand/week).

**Mitigation:** (a) Fall back to guidelines-derived weights for brands with < 50 feedback events. (b) Seed the classifier with synthetic labeled data from the Brand Profile Extractor (posts matching core themes = positive; off-limits matches = negative). (c) Prompt brand managers to rate alerts explicitly (thumbs up/down) during onboarding.

**Residual risk:** Medium. This is the most significant practical risk. Plan for a 4-6 week "warm-up period" where learning loops are in observation mode before activating parameter updates.

---

### Tradeoff 4: LLM Cost Scales with Brands × Alert Volume

**What we're optimizing for:** Alert quality and explanation richness. The LLM generates the "why relevant" explanation that brand managers rely on.

**What we're sacrificing:** Cost efficiency at scale. Adding brands increases LLM cost linearly.

**Mitigation:** The classifier pre-filter is the primary cost control — only top-20 posts/brand/cycle reach the LLM. At 40 brands, estimated ~$115/day. With 100 brands, ~$290/day, which remains within managed-service budgets. If cost becomes a concern, replace the LLM scorer with a fine-tuned smaller model after sufficient labeled data accumulates.

**Residual risk:** Low at current scale; medium if brand count reaches 100+.

---

### Tradeoff 5: Prompt Optimization is Semi-Manual

**What we're optimizing for:** Safety and auditability. Fully automated prompt updates risk introducing subtle regressions (the LLM that optimizes the prompt may optimize for superficial pattern-matching rather than true relevance improvement).

**What we're sacrificing:** Speed of iteration. The monthly review cycle means prompt improvements take weeks to deploy.

**Mitigation:** The A/B framework means even monthly prompt updates can be evaluated within 1 week of deployment before full rollout. Failure analysis is LLM-assisted, reducing manual burden to review + approval rather than full authorship.

**Residual risk:** Low. Prompt quality is less time-sensitive than query weights or thresholds, which update weekly/bi-weekly.

---

## Assumptions

1. The Social Listening API has sufficient rate limits for 40 brands × 48 queries/day = ~1,920 queries/day. If not, polling frequency is reduced to 60-minute intervals.
2. Brand managers are available to provide implicit feedback (clicks, dismissals) within hours of alert delivery. If managers only check dashboards daily, the "hours-not-days" requirement applies to alert *generation*, not action.
3. Brand guidelines PDFs are provided in text-extractable format (not scanned images). If scanned, add OCR as a pre-processing step.
4. The 2-3 engineer team has Python/ML proficiency. The stack (Postgres, S3, Python, managed LLM API) is intentionally conventional to minimize onboarding time.
5. Budget allows ~$115-200/day for LLM API calls. If budget is tighter, the system degrades gracefully: the classifier alone (no LLM) still delivers alerts, just without natural-language explanations.
