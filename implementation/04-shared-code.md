# Shared Code — Models, Schemas, Settings, Utils

## shared/settings.py

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://cpg:cpg@postgres:5432/cpgdb"
    database_sync_url: str = "postgresql://cpg:cpg@postgres:5432/cpgdb"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # S3 / MinIO
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_raw_posts_bucket: str = "raw-posts"
    s3_artifacts_bucket: str = "ml-artifacts"

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 512

    # Social API
    social_api_base_url: str = "http://mock-api:8001"
    social_api_key: str = "test-key"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Processing thresholds
    dedup_threshold: float = 0.97
    velocity_threshold: float = 1.5
    candidate_cosine_threshold: float = 0.55
    classifier_threshold: float = 0.65
    llm_score_threshold: float = 0.55

    # Alert limits
    max_alerts_per_brand_per_day: int = 10
    llm_candidates_per_cycle: int = 20

    # Scheduler
    ingest_interval_minutes: int = 30
```

## shared/db.py

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine
from shared.settings import Settings

settings = Settings()

# Async engine for FastAPI
async_engine = create_async_engine(settings.database_url, pool_size=20, max_overflow=10)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# Sync engine for Celery + Alembic
sync_engine = create_engine(settings.database_sync_url, pool_size=10)


async def get_db() -> AsyncSession:
    """FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        yield session
```

## shared/models.py

```python
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Brand(Base):
    __tablename__ = "brands"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: str = Column(String(255), nullable=False, unique=True)
    profile_json: dict = Column(JSONB, nullable=False, default={})
    # centroid column is vector(384) — managed via raw SQL, not ORM
    pdf_s3_key: str | None = Column(String(512), nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime | None = Column(DateTime(timezone=True), onupdate=func.now())

    query_terms = relationship("QueryTerm", back_populates="brand")
    alerts = relationship("Alert", back_populates="brand")


class QueryTerm(Base):
    __tablename__ = "query_terms"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    term: str = Column(String(255), nullable=False)
    term_type: str = Column(String(50), nullable=False)  # core | expansion | exclusion
    weight: float = Column(Float, nullable=False, default=1.0)
    source: str = Column(String(100), nullable=False)
    version: int = Column(Integer, nullable=False, default=1)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

    brand = relationship("Brand", back_populates="query_terms")


class Post(Base):
    __tablename__ = "posts"

    id: str = Column(String(255), primary_key=True)
    brand_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    platform: str = Column(String(50), nullable=False)
    text: str = Column(Text, nullable=False)
    author_id: str | None = Column(String(255), nullable=True)
    engagement_count: int = Column(Integer, nullable=False, default=0)
    published_at: datetime = Column(DateTime(timezone=True), nullable=False)
    ingested_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    s3_key: str | None = Column(String(512), nullable=True)
    meta: dict = Column(JSONB, nullable=False, default={})
    # embedding column is vector(384) — managed via raw SQL


class ClassifierParams(Base):
    __tablename__ = "classifier_params"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    version: int = Column(Integer, nullable=False)
    threshold: float = Column(Float, nullable=False, default=0.65)
    weights_s3_key: str | None = Column(String(512), nullable=True)
    f_score: float | None = Column(Float, nullable=True)
    precision_at_10: float | None = Column(Float, nullable=True)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: uuid.UUID | None = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    template_type: str = Column(String(100), nullable=False)
    version: int = Column(Integer, nullable=False)
    template_text: str = Column(Text, nullable=False)
    ab_weight: float = Column(Float, nullable=False, default=1.0)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    post_id: str = Column(String(255), ForeignKey("posts.id"), nullable=False)
    cluster_id: int | None = Column(Integer, nullable=True)
    relevance_score: float = Column(Float, nullable=False)
    velocity_score: float = Column(Float, nullable=False)
    engagement_score: float = Column(Float, nullable=False)
    novelty_score: float = Column(Float, nullable=False)
    composite_score: float = Column(Float, nullable=False)
    why_relevant: str | None = Column(Text, nullable=True)
    prompt_template_id: uuid.UUID | None = Column(UUID(as_uuid=True), ForeignKey("prompt_templates.id"), nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("brand_id", "post_id", name="uq_alert_brand_post"),)

    brand = relationship("Brand", back_populates="alerts")
    post = relationship("Post")


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=False)
    brand_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    action: str = Column(String(50), nullable=False)  # click | dismiss | content_brief | share
    dwell_ms: int | None = Column(Integer, nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
```

## shared/schemas.py

```python
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class BrandProfile(BaseModel):
    """Output of Brand Profile Extractor (LLM)."""
    brand_name: str
    category: str
    target_audience: str
    core_topics: list[str]
    tone_of_voice: str
    off_limits_topics: list[str]
    relevant_hashtags: list[str]
    competitor_brands: list[str]


class AlertOut(BaseModel):
    id: uuid.UUID
    brand_id: uuid.UUID
    post_id: str
    post_text: str
    platform: str
    composite_score: float
    relevance_score: float
    velocity_score: float
    why_relevant: str | None
    cluster_id: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackIn(BaseModel):
    alert_id: uuid.UUID
    action: str = Field(..., pattern="^(click|dismiss|content_brief|share)$")
    dwell_ms: int | None = None


class BrandOut(BaseModel):
    id: uuid.UUID
    name: str
    profile_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}
```

## shared/utils/embedder.py

```python
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from shared.settings import Settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = Settings()
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns (N, 384) array."""
    return _get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True)


def embed_posts(posts: List[Dict]) -> Dict[str, np.ndarray]:
    """Returns {post_id: 384-dim vector}."""
    model = _get_model()
    texts = [p["text"] for p in posts]
    ids = [p["id"] for p in posts]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return dict(zip(ids, embeddings))
```

## shared/utils/ngrams.py

```python
import re
from collections import Counter
from typing import List


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract word n-grams from text, lowercased, punctuation stripped."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(texts: List[str], n: int = 2, top_k: int = 20) -> List[tuple[str, int]]:
    """Return top-k n-grams by frequency across all texts."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(extract_ngrams(text, n))
    return counter.most_common(top_k)
```

## shared/utils/s3.py

```python
import json
import boto3
from botocore.client import Config
from shared.settings import Settings


def get_s3_client():
    settings = Settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )


def upload_json(bucket: str, key: str, data: dict) -> None:
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data), ContentType="application/json")


def download_json(bucket: str, key: str) -> dict:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def ensure_buckets_exist() -> None:
    settings = Settings()
    s3 = get_s3_client()
    for bucket in [settings.s3_raw_posts_bucket, settings.s3_artifacts_bucket]:
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            s3.create_bucket(Bucket=bucket)
```
