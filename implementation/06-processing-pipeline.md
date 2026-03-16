# Processing Pipeline — 5-Step ML Funnel (Celery Chain)

## Overview

Each ingest cycle triggers a Celery chain: `embed → dedup → velocity → candidate_filter → classify`.
~90% of posts are dropped before reaching the LLM scorer.

```
embed_posts_batch
      ↓
near_dedup_filter        (cosine > 0.97 → drop)
      ↓
velocity_score_filter    (growth_rate < 1.5 → drop)
      ↓
candidate_filter         (cosine to brand centroid < 0.55 → drop)
      ↓
classify_posts           (logistic regression < 0.65 → drop)
      ↓
llm_score (see 07-llm-and-alerts.md)
```

## services/worker/celery_app.py

```python
from celery import Celery
from shared.settings import Settings

settings = Settings()

celery_app = Celery(
    "cpg_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "services.worker.tasks.ingest",
        "services.worker.tasks.embed",
        "services.worker.tasks.process",
        "services.worker.tasks.llm_score",
        "services.worker.tasks.assemble",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)
```

## services/worker/tasks/embed.py

```python
"""
Step 1: Embed new posts and store vectors in Postgres via pgvector.
Step 2: Near-dedup — drop posts with cosine similarity > 0.97 to any existing post.
"""
import uuid
import numpy as np
import psycopg2
from celery import chain

from services.worker.celery_app import celery_app
from shared.settings import Settings
from shared.utils.embedder import embed_posts

settings = Settings()


def _get_conn():
    return psycopg2.connect(settings.database_sync_url)


@celery_app.task(name="embed.embed_posts_batch", bind=True, max_retries=3)
def embed_posts_batch(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Fetch post texts from DB, embed with sentence-transformers,
    store vectors in posts.embedding column.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM posts WHERE id = ANY(%s) AND brand_id = %s",
                (post_ids, uuid.UUID(brand_id)),
            )
            rows = cur.fetchall()

        posts = [{"id": row[0], "text": row[1]} for row in rows]
        if not posts:
            return {"embedded": 0}

        embeddings = embed_posts(posts)

        with conn.cursor() as cur:
            for post_id, emb in embeddings.items():
                vector_str = f"[{','.join(str(x) for x in emb.tolist())}]"
                cur.execute(
                    "UPDATE posts SET embedding = %s::vector WHERE id = %s",
                    (vector_str, post_id),
                )
        conn.commit()
    finally:
        conn.close()

    # Chain to dedup
    near_dedup_filter.delay(brand_id, post_ids)
    return {"embedded": len(embeddings)}


@celery_app.task(name="embed.near_dedup_filter", bind=True, max_retries=3)
def near_dedup_filter(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Drop posts with cosine similarity > 0.97 to any earlier post in the same brand.
    Idempotent: marks dropped posts with meta.dedup_dropped = true.
    """
    THRESHOLD = settings.dedup_threshold  # 0.97

    conn = _get_conn()
    surviving = []
    dropped = 0

    try:
        with conn.cursor() as cur:
            for post_id in post_ids:
                # Find this post's embedding
                cur.execute(
                    "SELECT embedding FROM posts WHERE id = %s", (post_id,)
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    continue

                embedding = row[0]

                # Check against all other brand posts (excluding itself)
                # cosine distance < (1 - THRESHOLD) means similarity > THRESHOLD
                cosine_distance_threshold = 1 - THRESHOLD
                cur.execute(
                    """
                    SELECT COUNT(*) FROM posts
                    WHERE brand_id = %s
                      AND id != %s
                      AND embedding IS NOT NULL
                      AND (embedding <=> %s::vector) < %s
                    """,
                    (uuid.UUID(brand_id), post_id, embedding, cosine_distance_threshold),
                )
                near_count = cur.fetchone()[0]

                if near_count > 0:
                    # Mark as duplicate
                    cur.execute(
                        "UPDATE posts SET meta = meta || '{\"dedup_dropped\": true}'::jsonb WHERE id = %s",
                        (post_id,),
                    )
                    dropped += 1
                else:
                    surviving.append(post_id)

        conn.commit()
    finally:
        conn.close()

    if surviving:
        from services.worker.tasks.process import velocity_score_filter
        velocity_score_filter.delay(brand_id, surviving)

    return {"surviving": len(surviving), "dropped": dropped}
```

## services/worker/tasks/process.py

```python
"""
Steps 3–5 of the processing funnel:
  3. Velocity Score Filter
  4. Candidate Filter (cosine similarity to brand centroid)
  5. Logistic Regression Classifier
"""
import uuid
import math
import json
import pickle
import numpy as np
import psycopg2
import redis
from datetime import datetime, timedelta

from services.worker.celery_app import celery_app
from shared.settings import Settings
from shared.utils.s3 import download_json

settings = Settings()
_redis_client = redis.from_url(settings.redis_url)


def _get_conn():
    return psycopg2.connect(settings.database_sync_url)


# ─── Step 3: Velocity Score ──────────────────────────────────────────────────

@celery_app.task(name="process.velocity_score_filter", bind=True, max_retries=3)
def velocity_score_filter(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Compute growth_rate = current_engagement / baseline_engagement.
    Baseline = median engagement for the same platform over the past 30 min window.
    Drop posts with growth_rate < 1.5.
    Velocity scores are cached in Redis for 2 hours.
    """
    THRESHOLD = settings.velocity_threshold  # 1.5

    conn = _get_conn()
    surviving = []
    dropped = 0

    try:
        with conn.cursor() as cur:
            # Get baselines per platform (cached in Redis)
            cur.execute(
                """
                SELECT platform, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY engagement_count) as median_eng
                FROM posts
                WHERE brand_id = %s
                  AND published_at > NOW() - INTERVAL '30 minutes'
                GROUP BY platform
                """,
                (uuid.UUID(brand_id),),
            )
            baselines = {row[0]: max(row[1], 1) for row in cur.fetchall()}

            for post_id in post_ids:
                cur.execute(
                    "SELECT platform, engagement_count FROM posts WHERE id = %s",
                    (post_id,),
                )
                row = cur.fetchone()
                if not row:
                    continue

                platform, eng = row
                baseline = baselines.get(platform, 1)
                growth_rate = eng / baseline

                # Cache velocity score
                _redis_client.setex(
                    f"velocity:{post_id}",
                    7200,  # 2h TTL
                    str(growth_rate),
                )

                if growth_rate >= THRESHOLD:
                    surviving.append(post_id)
                else:
                    dropped += 1
    finally:
        conn.close()

    if surviving:
        candidate_filter.delay(brand_id, surviving)

    return {"surviving": len(surviving), "dropped": dropped}


# ─── Step 4: Candidate Filter ────────────────────────────────────────────────

@celery_app.task(name="process.candidate_filter", bind=True, max_retries=3)
def candidate_filter(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Drop posts where cosine similarity to brand centroid < 0.55.
    This eliminates ~90% of ingested posts.
    Uses raw pgvector SQL — no ORM.
    """
    THRESHOLD = settings.candidate_cosine_threshold  # 0.55

    conn = _get_conn()
    surviving = []
    dropped = 0

    try:
        with conn.cursor() as cur:
            # Get brand centroid
            cur.execute(
                "SELECT centroid FROM brands WHERE id = %s",
                (uuid.UUID(brand_id),),
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                # No centroid yet — pass all through
                return {"surviving": len(post_ids), "dropped": 0}

            centroid = row[0]
            cosine_distance_threshold = 1 - THRESHOLD

            for post_id in post_ids:
                cur.execute(
                    """
                    SELECT (embedding <=> %s::vector) AS dist
                    FROM posts
                    WHERE id = %s AND embedding IS NOT NULL
                    """,
                    (centroid, post_id),
                )
                row = cur.fetchone()
                if row and row[0] < cosine_distance_threshold:
                    surviving.append(post_id)
                else:
                    dropped += 1
    finally:
        conn.close()

    if surviving:
        classify_posts.delay(brand_id, surviving)

    return {"surviving": len(surviving), "dropped": dropped}


# ─── Step 5: Logistic Regression Classifier ──────────────────────────────────

def _build_features(post_id: str, brand_id: str, conn) -> np.ndarray | None:
    """
    Build 12-feature vector for logistic regression classifier.
    Features:
      0-3:  text stats (char_count, word_count, has_url, has_hashtag)
      4-5:  engagement (log_engagement, engagement_percentile)
      6:    velocity_score (from Redis cache)
      7-9:  embedding stats (cosine_to_centroid, embedding_norm, embedding_variance)
      10:   platform_encoded (tiktok=0, instagram=1, twitter=2, other=3)
      11:   hour_of_day (0-23)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.text, p.engagement_count, p.platform, p.published_at,
                   (p.embedding <=> b.centroid) AS centroid_dist,
                   p.embedding
            FROM posts p
            JOIN brands b ON b.id = p.brand_id
            WHERE p.id = %s AND p.embedding IS NOT NULL
            """,
            (post_id,),
        )
        row = cur.fetchone()

    if not row:
        return None

    text, eng, platform, pub_at, centroid_dist, embedding = row

    velocity = float(_redis_client.get(f"velocity:{post_id}") or 1.0)

    platform_map = {"tiktok": 0, "instagram": 1, "twitter": 2}
    platform_enc = platform_map.get(platform, 3)

    # Parse embedding from pgvector string format "[0.1,0.2,...]"
    emb_arr = np.array([float(x) for x in embedding.strip("[]").split(",")])
    emb_norm = float(np.linalg.norm(emb_arr))
    emb_var = float(np.var(emb_arr))

    features = np.array([
        len(text),                              # 0: char_count
        len(text.split()),                      # 1: word_count
        1.0 if "http" in text else 0.0,         # 2: has_url
        1.0 if "#" in text else 0.0,            # 3: has_hashtag
        math.log(eng + 1),                      # 4: log_engagement
        min(eng / 10000, 1.0),                  # 5: engagement_percentile (rough)
        velocity,                               # 6: velocity_score
        1 - centroid_dist,                      # 7: cosine_to_centroid (similarity)
        emb_norm,                               # 8: embedding_norm
        emb_var,                                # 9: embedding_variance
        float(platform_enc),                    # 10: platform_encoded
        float(pub_at.hour),                     # 11: hour_of_day
    ], dtype=np.float32)

    return features


def _load_classifier(brand_id: str):
    """Load sklearn LogisticRegression from S3. Returns None if not trained yet."""
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )
    key = f"classifiers/{brand_id}/latest.pkl"
    try:
        response = s3.get_object(Bucket=settings.s3_artifacts_bucket, Key=key)
        return pickle.loads(response["Body"].read())
    except Exception:
        return None


@celery_app.task(name="process.classify_posts", bind=True, max_retries=3)
def classify_posts(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Score posts with logistic regression classifier.
    Falls back to cosine-similarity-only scoring if no model trained yet.
    Drops posts with score < classifier_threshold.
    """
    THRESHOLD = settings.classifier_threshold  # 0.65

    classifier = _load_classifier(brand_id)
    conn = _get_conn()
    surviving = []
    dropped = 0

    try:
        for post_id in post_ids:
            features = _build_features(post_id, brand_id, conn)
            if features is None:
                dropped += 1
                continue

            if classifier is not None:
                score = float(classifier.predict_proba([features])[0][1])
            else:
                # Fallback: use cosine similarity to centroid as score
                score = features[7]  # cosine_to_centroid

            if score >= THRESHOLD:
                surviving.append((post_id, score))
            else:
                dropped += 1
    finally:
        conn.close()

    if surviving:
        from services.worker.tasks.llm_score import llm_score_batch
        post_ids_surviving = [pid for pid, _ in surviving]
        llm_score_batch.delay(brand_id, post_ids_surviving)

    return {"surviving": len(surviving), "dropped": dropped}
```
