# Database Schema — PostgreSQL 16 + pgvector

## migrations/env.py

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from shared.models import Base
from shared.settings import Settings

config = context.config
settings = Settings()

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = settings.database_sync_url
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = settings.database_sync_url
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## migrations/versions/0001_initial.py

```python
"""Initial schema

Revision ID: 0001
Create Date: 2024-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── brands ──────────────────────────────────────────────────────────────
    op.create_table(
        "brands",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("profile_json", JSONB, nullable=False, default={}),
        # 384-dim centroid from brand guidelines PDF
        sa.Column("embedding_centroid", sa.Text, nullable=True),  # stored as pgvector
        sa.Column("pdf_s3_key", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    # Add vector column separately (pgvector DDL)
    op.execute("ALTER TABLE brands ADD COLUMN IF NOT EXISTS centroid vector(384)")

    # ── query_terms ──────────────────────────────────────────────────────────
    # Versioned: never delete, only add new rows
    op.create_table(
        "query_terms",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=False),
        sa.Column("term", sa.String(255), nullable=False),
        sa.Column("term_type", sa.String(50), nullable=False),  # core | expansion | exclusion
        sa.Column("weight", sa.Float, nullable=False, default=1.0),
        sa.Column("source", sa.String(100), nullable=False),  # manual | llm_extract | learning_loop
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_query_terms_brand_active", "query_terms", ["brand_id", "is_active"])

    # ── posts ────────────────────────────────────────────────────────────────
    op.create_table(
        "posts",
        sa.Column("id", sa.String(255), primary_key=True),  # external post_id from Social API
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=False),
        sa.Column("platform", sa.String(50), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("author_id", sa.String(255), nullable=True),
        sa.Column("engagement_count", sa.Integer, nullable=False, default=0),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("s3_key", sa.String(512), nullable=True),
        sa.Column("meta", JSONB, nullable=False, default={}),
    )
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS embedding vector(384)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_posts_embedding "
        "ON posts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.create_index("ix_posts_brand_published", "posts", ["brand_id", "published_at"])

    # ── classifier_params ────────────────────────────────────────────────────
    # Versioned: never delete old versions
    op.create_table(
        "classifier_params",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False, default=0.65),
        sa.Column("weights_s3_key", sa.String(512), nullable=True),
        sa.Column("f_score", sa.Float, nullable=True),
        sa.Column("precision_at_10", sa.Float, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── prompt_templates ─────────────────────────────────────────────────────
    # Versioned: never delete old versions
    op.create_table(
        "prompt_templates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=True),
        sa.Column("template_type", sa.String(100), nullable=False),  # llm_score | suppression
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("template_text", sa.Text, nullable=False),
        sa.Column("ab_weight", sa.Float, nullable=False, default=1.0),  # 0.0–1.0 traffic split
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── alerts ───────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=False),
        sa.Column("post_id", sa.String(255), sa.ForeignKey("posts.id"), nullable=False),
        sa.Column("cluster_id", sa.Integer, nullable=True),  # HDBSCAN label
        sa.Column("relevance_score", sa.Float, nullable=False),
        sa.Column("velocity_score", sa.Float, nullable=False),
        sa.Column("engagement_score", sa.Float, nullable=False),
        sa.Column("novelty_score", sa.Float, nullable=False),
        sa.Column("composite_score", sa.Float, nullable=False),
        sa.Column("why_relevant", sa.Text, nullable=True),
        sa.Column("prompt_template_id", UUID(as_uuid=True), sa.ForeignKey("prompt_templates.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("brand_id", "post_id", name="uq_alert_brand_post"),
    )
    op.create_index("ix_alerts_brand_created", "alerts", ["brand_id", "created_at"])

    # ── feedback_events ───────────────────────────────────────────────────────
    op.create_table(
        "feedback_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("alert_id", UUID(as_uuid=True), sa.ForeignKey("alerts.id"), nullable=False),
        sa.Column("brand_id", UUID(as_uuid=True), sa.ForeignKey("brands.id"), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),  # click | dismiss | content_brief | share
        sa.Column("dwell_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_brand_created", "feedback_events", ["brand_id", "created_at"])


def downgrade() -> None:
    op.drop_table("feedback_events")
    op.drop_table("alerts")
    op.drop_table("prompt_templates")
    op.drop_table("classifier_params")
    op.drop_table("posts")
    op.drop_table("query_terms")
    op.drop_table("brands")
```

## Notes on pgvector Usage

- **Never use ORM** for vector operations — use raw SQL with `asyncpg` or `psycopg2`
- Vector column type: `vector(384)` for all-MiniLM-L6-v2 output
- Cosine similarity query pattern:
  ```sql
  SELECT id, text, (embedding <=> $1::vector) AS distance
  FROM posts
  WHERE brand_id = $2
    AND (embedding <=> $1::vector) < 0.45
  ORDER BY distance
  LIMIT 20;
  ```
- The `<=>` operator = cosine distance (1 − similarity); `<->` = L2 distance
- IVFFlat index with `lists=100` for ~40 brands × many posts; use `HNSW` if post count exceeds 1M
