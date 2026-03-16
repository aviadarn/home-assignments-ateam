# Tech Stack — Exact Versions

## Python Services

```
Python                  3.12
FastAPI                 0.115.0
uvicorn[standard]       0.30.0
pydantic                2.7.0
pydantic-settings       2.3.0
SQLAlchemy              2.0.35
alembic                 1.13.2
asyncpg                 0.29.0
psycopg2-binary         2.9.9        # sync: Alembic + Celery
redis                   5.0.8
celery[redis]           5.4.0
APScheduler             3.10.4
httpx                   0.27.0       # async HTTP for Social API + Anthropic
anthropic               0.34.0       # Claude Haiku via Bedrock
boto3                   1.35.0       # S3 / MinIO
sentence-transformers   3.1.0        # all-MiniLM-L6-v2
hdbscan                 0.8.38.post1
scikit-learn            1.5.2
numpy                   1.26.4
PyPDF2                  3.0.1        # brand PDF parsing
python-multipart        0.0.9        # file upload
websockets              13.0
```

## Databases & Infrastructure

```
PostgreSQL              16
pgvector extension      0.7.4
Redis                   7.4
MinIO                   RELEASE.2024-09-22  # S3-compatible local dev
```

## Docker Base Images

```
python:3.12-slim        # all service Dockerfiles
postgres:16             # with pgvector installed at startup
redis:7.4-alpine
minio/minio:latest
```

## Dev Tooling

```
pytest                  8.3.0
pytest-asyncio          0.23.8
httpx                   0.27.0       # TestClient
factory-boy             3.3.1
black                   24.8.0
ruff                    0.6.0
mypy                    1.11.0
```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "cpg-trend-detection"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.0",
    "pydantic==2.7.0",
    "pydantic-settings==2.3.0",
    "sqlalchemy==2.0.35",
    "alembic==1.13.2",
    "asyncpg==0.29.0",
    "psycopg2-binary==2.9.9",
    "redis==5.0.8",
    "celery[redis]==5.4.0",
    "apscheduler==3.10.4",
    "httpx==0.27.0",
    "anthropic==0.34.0",
    "boto3==1.35.0",
    "sentence-transformers==3.1.0",
    "hdbscan==0.8.38.post1",
    "scikit-learn==1.5.2",
    "numpy==1.26.4",
    "PyPDF2==3.0.1",
    "python-multipart==0.0.9",
    "websockets==13.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.0",
    "pytest-asyncio==0.23.8",
    "httpx==0.27.0",
    "factory-boy==3.3.1",
    "black==24.8.0",
    "ruff==0.6.0",
    "mypy==1.11.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

## Makefile

```makefile
.PHONY: up down build migrate seed test lint

up:
	docker compose up -d

down:
	docker compose down -v

build:
	docker compose build

migrate:
	docker compose run --rm worker alembic upgrade head

seed:
	docker compose run --rm worker python scripts/seed_brands.py

test:
	docker compose run --rm worker pytest tests/ -v

lint:
	ruff check .
	black --check .
	mypy shared/ services/

logs:
	docker compose logs -f

shell-db:
	docker compose exec postgres psql -U cpg -d cpgdb
```
