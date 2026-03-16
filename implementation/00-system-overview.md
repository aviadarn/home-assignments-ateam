# System Overview — CPG Social Trend Detection

## What This System Does

Monitors social media in real-time for 40+ CPG brands (e.g., Pepsi, Lay's, Gatorade under a single conglomerate). Every 30 minutes it:
1. Queries a Social Listening API using boolean search queries per brand
2. Runs ingested posts through a 5-step ML processing funnel
3. Scores surviving posts with an LLM and assembles trend alerts
4. Delivers alerts to brand managers via REST API and WebSocket push
5. Runs 3 learning loops (weekly/bi-weekly/monthly) to improve precision over time

## Critical Constraint

The Social Listening API is **query-based** — it requires explicit boolean search parameters. You cannot pull a raw feed. This shapes every architectural decision: we must maintain a library of well-tuned query terms per brand, and continuously improve them via the Query Expansion learning loop.

## End-to-End Data Flow

```
PDF Brand Guidelines
        ↓
  Brand Profile Extractor (Claude)
        ↓ structured JSON + 384-dim centroid
  Boolean Query Generator
        ↓ core terms + expansion terms (weight > 0.6)
  Social Listening API (polled every 30 min)
        ↓ raw posts → S3
  ┌─────────────────────────────────────┐
  │  5-Step Processing Funnel (Celery)  │
  │  1. Embed (sentence-transformers)   │
  │  2. Near-Dedup (cosine > 0.97)      │
  │  3. Velocity Score (growth > 1.5x)  │
  │  4. Candidate Filter (cos > 0.55)   │
  │  5. Logistic Classifier (p > 0.65)  │
  └─────────────────────────────────────┘
        ↓ ~10% survive funnel
  LLM Scorer (Claude Haiku via Bedrock)
        ↓ relevance score + why_relevant text
  Alert Assembler (HDBSCAN clusters)
        ↓ composite score, max 10 alerts/brand/day
  REST API + WebSocket → Brand Manager Dashboard
        ↓
  Feedback Events (click / dismiss / content_brief / share)
        ↓
  3 Learning Loops (APScheduler)
```

## Component Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| Brand Extractor | `services/worker/tasks/ingest.py` | PDF → structured brand profile |
| Query Generator | `services/worker/tasks/ingest.py` | Brand profile → boolean queries |
| Post Ingester | `services/scheduler/jobs/query_runner.py` | Polls Social API every 30 min |
| Embedder | `shared/utils/embedder.py` | all-MiniLM-L6-v2, 384 dims |
| Near-Dedup | `services/worker/tasks/embed.py` | Cosine similarity dedup |
| Velocity Scorer | `services/worker/tasks/process.py` | Growth rate vs 30-min baseline |
| Candidate Filter | `services/worker/tasks/process.py` | Cosine similarity vs brand centroid |
| Logistic Classifier | `services/worker/tasks/process.py` | 12-feature relevance classifier |
| LLM Scorer | `services/worker/tasks/llm_score.py` | Claude Haiku scoring + explanation |
| Alert Assembler | `services/worker/tasks/assemble.py` | HDBSCAN + composite score |
| Dashboard API | `services/api/` | FastAPI REST + WebSocket |
| Learning Loops | `services/scheduler/jobs/` | 3 improvement loops |

## Key Numbers

- 40 brands × 48 queries/day = 1,920 Social API calls/day
- Funnel drops ~90% of posts (10% reach LLM scoring)
- LLM scoring: top 20 posts/brand/cycle × 40 brands × 48 cycles = 38,400 calls/day ≈ $115/day
- Max alerts: 10/brand/day
- Worst-case latency: 37 minutes (post published → alert delivered)
- CTR target: 12% baseline → 18% after 8 weeks of learning loops

## Storage Architecture

| Data | Store | Reason |
|------|-------|--------|
| Raw posts (JSON) | S3 (`raw-posts/`) | Cheap, durable, unlimited |
| Post embeddings | pgvector (Postgres) | Vector similarity search |
| Brand centroids | pgvector (Postgres) | Brand isolation enforced at query level |
| Velocity cache | Redis | Sub-millisecond read, TTL 2h |
| ML model artifacts | S3 (`ml-artifacts/`) | Versioned by timestamp |
| All operational data | Postgres | Alerts, feedback, query terms, params |
