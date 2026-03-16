# CPG Social Trend Detection — Implementation Guide for Claude Code

## How to Use These Files

Read files **in order** (00 → 10). Each file is self-contained and builds on the previous.
Before writing any code, read all files once to understand the full system.

## What You Are Building

A production-ready Python/FastAPI backend that:
1. Reads brand guidelines PDFs → extracts structured profiles (LLM)
2. Generates boolean search queries per brand → polls a Social Listening API every 30 min
3. Runs ingested posts through a 5-step ML processing funnel
4. Scores final candidates with an LLM → assembles trend alerts
5. Serves alerts to brand managers via REST API + WebSocket
6. Runs 3 learning loops (weekly, bi-weekly, monthly) to improve over time

## Implementation Order

```
Step 1 → 02-docker-compose.md    → docker-compose.yml + .env.example
Step 2 → 03-database-schema.md   → Postgres tables + Alembic migrations
Step 3 → 04-shared-code.md       → SQLAlchemy models, Pydantic schemas, shared utils
Step 4 → 05-ingestion.md         → Brand extractor + Query generator + Post ingester
Step 5 → 06-processing-pipeline.md → 5-step ML funnel (Celery chain)
Step 6 → 07-llm-and-alerts.md    → LLM scorer + Alert assembler
Step 7 → 08-dashboard-api.md     → FastAPI REST API + WebSocket
Step 8 → 09-learning-loops.md    → 3 learning loops (APScheduler)
Step 9 → 10-testing.md           → Tests + verification
```

## Project Root Structure to Create

```
cpg-trend-detection/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── .env                          ← create from .env.example, never commit
├── Makefile
├── pyproject.toml
├── alembic.ini
├── migrations/
│   └── versions/
├── shared/                       ← shared Python package (models, schemas, db)
│   ├── __init__.py
│   ├── db.py
│   ├── models.py                 ← SQLAlchemy ORM models
│   ├── schemas.py                ← Pydantic v2 schemas
│   ├── settings.py               ← Pydantic Settings (reads .env)
│   └── utils/
│       ├── embedder.py
│       ├── ngrams.py
│       └── s3.py
├── services/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── alerts.py
│   │   │   ├── feedback.py
│   │   │   ├── brands.py
│   │   │   └── ws.py
│   │   └── requirements.txt
│   ├── worker/
│   │   ├── Dockerfile
│   │   ├── celery_app.py
│   │   ├── tasks/
│   │   │   ├── ingest.py
│   │   │   ├── embed.py
│   │   │   ├── process.py
│   │   │   ├── llm_score.py
│   │   │   └── assemble.py
│   │   └── requirements.txt
│   └── scheduler/
│       ├── Dockerfile
│       ├── main.py               ← APScheduler entry point
│       ├── jobs/
│       │   ├── query_runner.py   ← runs every 30 min per brand
│       │   ├── loop_expansion.py ← weekly Query Expansion loop
│       │   ├── loop_threshold.py ← bi-weekly Threshold Calibration loop
│       │   └── loop_prompt.py    ← monthly Prompt Optimization loop
│       └── requirements.txt
├── scripts/
│   ├── seed_brands.py            ← load sample brand profiles
│   ├── mock_social_api.py        ← local mock of Social Listening API
│   └── run_demo.py               ← end-to-end demo
└── tests/
    ├── conftest.py
    ├── test_processing.py
    ├── test_learning_loops.py
    └── test_api.py
```

## Key Constraints to Respect

1. **No ORM magic for vector operations** — use raw SQL with `psycopg2` or `asyncpg` for pgvector queries (`<->`, `<=>` operators)
2. **All service settings via environment variables** — use `pydantic-settings` with a `Settings` class
3. **Idempotent tasks** — all Celery tasks must be safe to re-run (dedup by post_id)
4. **Versioned parameters** — query_terms, classifier thresholds, and prompt templates all have version columns; never delete old versions, only add new ones
5. **Brand isolation** — no shared state between brands in any query or computation
6. **Async FastAPI** — all API endpoints use `async def` with `asyncpg` connection pool

## Environment Variables Required

```bash
DATABASE_URL=postgresql+asyncpg://cpg:cpg@postgres:5432/cpgdb
DATABASE_SYNC_URL=postgresql://cpg:cpg@postgres:5432/cpgdb  # for Alembic + Celery
REDIS_URL=redis://redis:6379/0
S3_ENDPOINT=http://minio:9000                                # or AWS endpoint
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_RAW_POSTS_BUCKET=raw-posts
S3_ARTIFACTS_BUCKET=ml-artifacts
ANTHROPIC_API_KEY=sk-ant-...                                 # for LLM scoring
SOCIAL_API_BASE_URL=http://mock-api:8001                     # or real API URL
SOCIAL_API_KEY=test-key
EMBEDDING_MODEL=all-MiniLM-L6-v2
```
