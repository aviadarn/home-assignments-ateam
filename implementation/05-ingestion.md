# Ingestion — Brand Extractor + Query Generator + Post Ingester

## services/worker/tasks/ingest.py

```python
"""
Brand Profile Extractor, Query Generator, and Post Ingester.

All tasks are idempotent — safe to re-run (dedup by post_id).
"""
import json
import uuid
import hashlib
from datetime import datetime
import httpx
import PyPDF2
import io
import numpy as np
from anthropic import Anthropic
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from services.worker.celery_app import celery_app
from shared.models import Brand, QueryTerm, Post
from shared.schemas import BrandProfile
from shared.settings import Settings
from shared.utils.embedder import embed_texts
from shared.utils.s3 import upload_json, ensure_buckets_exist
from shared.db import sync_engine

settings = Settings()
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)


# ─── Brand Profile Extractor ─────────────────────────────────────────────────

@celery_app.task(name="ingest.extract_brand_profile", bind=True, max_retries=3)
def extract_brand_profile(self, brand_id: str, pdf_s3_key: str) -> dict:
    """
    Extract structured brand profile from a PDF using Claude.
    Stores result in brands.profile_json and computes embedding centroid.
    """
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )

    # Download PDF from S3
    response = s3.get_object(Bucket=settings.s3_artifacts_bucket, Key=pdf_s3_key)
    pdf_bytes = response["Body"].read()

    # Extract text from PDF
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)[:8000]

    # LLM extraction
    prompt = f"""Extract a structured brand profile from this brand guidelines document.
Return ONLY valid JSON with these exact fields:
- brand_name: string
- category: string (e.g. "beverages", "snacks", "personal care")
- target_audience: string
- core_topics: list of 5-10 strings (topics this brand cares about)
- tone_of_voice: string
- off_limits_topics: list of strings (topics to exclude)
- relevant_hashtags: list of strings (without #)
- competitor_brands: list of strings

Document:
{pdf_text}"""

    message = anthropic_client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    profile_data = json.loads(message.content[0].text)
    profile = BrandProfile(**profile_data)

    # Compute embedding centroid from core_topics + relevant_hashtags
    texts_to_embed = profile.core_topics + profile.relevant_hashtags
    embeddings = embed_texts(texts_to_embed)
    centroid = embeddings.mean(axis=0).tolist()

    # Persist to database
    with Session(sync_engine) as session:
        brand = session.get(Brand, uuid.UUID(brand_id))
        if brand is None:
            raise ValueError(f"Brand {brand_id} not found")
        brand.profile_json = profile.model_dump()
        session.flush()

        # Store centroid as pgvector via raw SQL
        session.execute(
            "UPDATE brands SET centroid = :centroid::vector WHERE id = :id",
            {"centroid": f"[{','.join(str(x) for x in centroid)}]", "id": brand_id},
        )

        # Seed initial query terms
        _seed_query_terms(session, brand_id, profile)
        session.commit()

    return profile.model_dump()


def _seed_query_terms(session: Session, brand_id: str, profile: BrandProfile) -> None:
    """Insert core terms and exclusions. Idempotent — skips existing terms."""
    existing = {
        row.term
        for row in session.execute(
            select(QueryTerm.term).where(
                and_(QueryTerm.brand_id == uuid.UUID(brand_id), QueryTerm.is_active == True)
            )
        )
    }

    terms_to_add = []

    for topic in profile.core_topics:
        if topic not in existing:
            terms_to_add.append(QueryTerm(
                brand_id=uuid.UUID(brand_id),
                term=topic,
                term_type="core",
                weight=1.0,
                source="llm_extract",
                version=1,
            ))

    for hashtag in profile.relevant_hashtags:
        if hashtag not in existing:
            terms_to_add.append(QueryTerm(
                brand_id=uuid.UUID(brand_id),
                term=hashtag,
                term_type="core",
                weight=0.8,
                source="llm_extract",
                version=1,
            ))

    for topic in profile.off_limits_topics:
        if topic not in existing:
            terms_to_add.append(QueryTerm(
                brand_id=uuid.UUID(brand_id),
                term=topic,
                term_type="exclusion",
                weight=1.0,
                source="llm_extract",
                version=1,
            ))

    session.add_all(terms_to_add)


# ─── Query Generator ─────────────────────────────────────────────────────────

def build_boolean_query(brand_id: str) -> str:
    """
    Build a boolean search query string for the Social Listening API.
    Format: (core1 OR core2 OR exp1) NOT (excl1 OR excl2)
    Only includes expansion terms with weight > 0.6.
    """
    with Session(sync_engine) as session:
        terms = session.execute(
            select(QueryTerm).where(
                and_(
                    QueryTerm.brand_id == uuid.UUID(brand_id),
                    QueryTerm.is_active == True,
                )
            )
        ).scalars().all()

    include_terms = []
    exclude_terms = []

    for qt in terms:
        if qt.term_type == "exclusion":
            exclude_terms.append(qt.term)
        elif qt.term_type == "core":
            include_terms.append(qt.term)
        elif qt.term_type == "expansion" and qt.weight > 0.6:
            include_terms.append(qt.term)

    if not include_terms:
        raise ValueError(f"No active include terms for brand {brand_id}")

    query = "(" + " OR ".join(f'"{t}"' for t in include_terms) + ")"
    if exclude_terms:
        query += " NOT (" + " OR ".join(f'"{t}"' for t in exclude_terms) + ")"

    return query


# ─── Post Ingester ────────────────────────────────────────────────────────────

@celery_app.task(name="ingest.ingest_posts_for_brand", bind=True, max_retries=3)
def ingest_posts_for_brand(self, brand_id: str) -> dict:
    """
    Poll Social Listening API for new posts matching the brand's boolean query.
    Deduplicates by post_id. Stores raw JSON to S3 and metadata to Postgres.
    Returns {brand_id, new_posts, skipped_posts}.
    """
    ensure_buckets_exist()

    query = build_boolean_query(brand_id)

    # Call Social API
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{settings.social_api_base_url}/v1/search",
            params={"q": query, "limit": 200, "since": "30m"},
            headers={"X-API-Key": settings.social_api_key},
        )
        response.raise_for_status()
        raw_posts = response.json()["posts"]

    new_count = 0
    skipped_count = 0
    post_ids_to_process = []

    with Session(sync_engine) as session:
        for raw in raw_posts:
            post_id = raw["id"]

            # Idempotent: skip if already ingested
            existing = session.get(Post, post_id)
            if existing:
                skipped_count += 1
                continue

            # Store raw JSON to S3
            s3_key = f"brands/{brand_id}/posts/{post_id}.json"
            upload_json(settings.s3_raw_posts_bucket, s3_key, raw)

            # Insert post record
            post = Post(
                id=post_id,
                brand_id=uuid.UUID(brand_id),
                platform=raw.get("platform", "unknown"),
                text=raw["text"],
                author_id=raw.get("author_id"),
                engagement_count=raw.get("engagement_count", 0),
                published_at=datetime.fromisoformat(raw["published_at"]),
                s3_key=s3_key,
                meta=raw.get("meta", {}),
            )
            session.add(post)
            post_ids_to_process.append(post_id)
            new_count += 1

        session.commit()

    # Kick off processing pipeline for new posts
    if post_ids_to_process:
        from services.worker.tasks.embed import embed_posts_batch
        embed_posts_batch.delay(brand_id, post_ids_to_process)

    return {
        "brand_id": brand_id,
        "new_posts": new_count,
        "skipped_posts": skipped_count,
        "pipeline_triggered": len(post_ids_to_process) > 0,
    }
```

## scripts/mock_social_api.py

```python
"""
Local mock of the Social Listening API.
Serves GET /v1/search?q=...&limit=N&since=Xm
Returns synthetic posts matching the query.
"""
import random
import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
import uvicorn

app = FastAPI()

SAMPLE_POSTS = [
    "Just tried the new morning wellness routine with my favorite energy drink 🌅 #wellness #morningroutine",
    "5am club is real — productivity is through the roof when you start early",
    "Deep work session complete. 4 hours of uninterrupted focus. Game changer.",
    "Healthy snacking is underrated. My afternoon pick-me-up keeps me going all day",
    "New flavor just dropped and it tastes incredible 🔥 #newproduct",
    "Why does everyone sleep on hydration? It literally changes everything",
    "Gym session fueled by my go-to pre-workout. Never missing a beat.",
    "The rise of functional beverages is insane rn. Everyone is drinking their vitamins",
    "Morning routine content is taking over TikTok and honestly I'm here for it",
    "Snack haul from Costco just hit different this week ngl",
]


@app.get("/v1/search")
def search_posts(
    q: str = Query(...),
    limit: int = Query(50),
    since: str = Query("30m"),
):
    count = random.randint(10, min(limit, 80))
    posts = []
    for _ in range(count):
        posts.append({
            "id": str(uuid.uuid4()),
            "text": random.choice(SAMPLE_POSTS),
            "platform": random.choice(["tiktok", "instagram", "twitter"]),
            "author_id": f"user_{random.randint(1000, 9999)}",
            "engagement_count": random.randint(10, 50000),
            "published_at": (
                datetime.utcnow() - timedelta(minutes=random.randint(1, 29))
            ).isoformat(),
            "meta": {"query": q},
        })
    return {"posts": posts, "total": count}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```
