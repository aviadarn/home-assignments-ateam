# Docker Compose — Full Configuration

## docker-compose.yml

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: cpg
      POSTGRES_PASSWORD: cpg
      POSTGRES_DB: cpgdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-pgvector.sql:/docker-entrypoint-initdb.d/init-pgvector.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cpg -d cpgdb"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7.4-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

  mock-api:
    build:
      context: .
      dockerfile: scripts/Dockerfile.mock
    ports:
      - "8001:8001"
    environment:
      - MOCK_POST_COUNT=50
    command: python scripts/mock_social_api.py

  api:
    build:
      context: .
      dockerfile: services/api/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./shared:/app/shared
      - ./services/api:/app/services/api
    command: uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build:
      context: .
      dockerfile: services/worker/Dockerfile
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    volumes:
      - ./shared:/app/shared
      - ./services/worker:/app/services/worker
    command: celery -A services.worker.celery_app worker --loglevel=info --concurrency=4

  scheduler:
    build:
      context: .
      dockerfile: services/scheduler/Dockerfile
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      worker:
        condition: service_started
    volumes:
      - ./shared:/app/shared
      - ./services/scheduler:/app/services/scheduler
    command: python services/scheduler/main.py

  flower:
    image: mher/flower:2.0
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  postgres_data:
  minio_data:
```

## .env.example

```bash
# Database
DATABASE_URL=postgresql+asyncpg://cpg:cpg@postgres:5432/cpgdb
DATABASE_SYNC_URL=postgresql://cpg:cpg@postgres:5432/cpgdb

# Redis
REDIS_URL=redis://redis:6379/0

# S3 / MinIO
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_RAW_POSTS_BUCKET=raw-posts
S3_ARTIFACTS_BUCKET=ml-artifacts

# LLM
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-haiku-4-5-20251001
LLM_MAX_TOKENS=512

# Social API
SOCIAL_API_BASE_URL=http://mock-api:8001
SOCIAL_API_KEY=test-key

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Processing thresholds (can be tuned by learning loops)
DEDUP_THRESHOLD=0.97
VELOCITY_THRESHOLD=1.5
CANDIDATE_COSINE_THRESHOLD=0.55
CLASSIFIER_THRESHOLD=0.65
LLM_SCORE_THRESHOLD=0.55

# Alert limits
MAX_ALERTS_PER_BRAND_PER_DAY=10
LLM_CANDIDATES_PER_CYCLE=20

# Scheduler
INGEST_INTERVAL_MINUTES=30
EXPANSION_LOOP_DAY=monday
THRESHOLD_LOOP_DAY=monday
THRESHOLD_LOOP_WEEK=2
PROMPT_LOOP_DAY=1
```

## scripts/init-pgvector.sql

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## services/api/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY shared/ ./shared/
COPY services/api/ ./services/api/

ENV PYTHONPATH=/app
```

## services/worker/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# hdbscan needs build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY shared/ ./shared/
COPY services/worker/ ./services/worker/
COPY scripts/ ./scripts/
COPY migrations/ ./migrations/
COPY alembic.ini .

ENV PYTHONPATH=/app
```

## services/scheduler/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY shared/ ./shared/
COPY services/scheduler/ ./services/scheduler/

ENV PYTHONPATH=/app
```

## alembic.ini

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = %(DATABASE_SYNC_URL)s

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```
