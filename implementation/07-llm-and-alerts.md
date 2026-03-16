# LLM Scorer + Alert Assembler

## Overview

After the 5-step funnel, surviving posts go to:
1. **LLM Scorer** — Claude Haiku scores top 20 posts/brand/cycle for relevance + generates `why_relevant` text
2. **Alert Assembler** — HDBSCAN clusters posts, computes composite score, enforces max 10 alerts/brand/day

**Composite score formula:**
```
composite = 0.35 × relevance_score
          + 0.35 × velocity_score_normalized
          + 0.20 × log(engagement_count + 1) / log(max_engagement + 1)
          + 0.10 × novelty_score
```

## services/worker/tasks/llm_score.py

```python
"""
LLM Scorer: scores top 20 posts per brand per cycle using Claude Haiku.
Generates relevance score (0-1) and why_relevant explanation.
Cost: ~$115/day for 40 brands × 48 cycles × 20 posts.
"""
import uuid
import json
import psycopg2
from anthropic import Anthropic
from sqlalchemy.orm import Session

from services.worker.celery_app import celery_app
from shared.models import PromptTemplate
from shared.settings import Settings
from shared.db import sync_engine

settings = Settings()
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)


def _get_conn():
    return psycopg2.connect(settings.database_sync_url)


def _get_active_prompt(brand_id: str, template_type: str = "llm_score") -> tuple[str, str | None]:
    """
    Returns (template_text, template_id) for the active prompt template.
    Respects A/B traffic split via ab_weight.
    """
    import random

    with Session(sync_engine) as session:
        templates = session.execute(
            """
            SELECT id, template_text, ab_weight
            FROM prompt_templates
            WHERE (brand_id = :brand_id OR brand_id IS NULL)
              AND template_type = :template_type
              AND is_active = true
            ORDER BY brand_id DESC NULLS LAST
            """,
            {"brand_id": brand_id, "template_type": template_type},
        ).fetchall()

    if not templates:
        return _default_llm_score_prompt(), None

    # Weighted random selection for A/B test
    total_weight = sum(t[2] for t in templates)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for template_id, template_text, weight in templates:
        cumulative += weight
        if r <= cumulative:
            return template_text, str(template_id)

    return templates[0][1], str(templates[0][0])


def _default_llm_score_prompt() -> str:
    return """You are evaluating whether a social media post is relevant to a CPG brand.

Brand: {brand_name}
Brand category: {category}
Target audience: {target_audience}
Core topics: {core_topics}

Post text:
{post_text}

Score this post's relevance to the brand on a scale from 0.0 to 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat related to the category
- 1.0 = highly relevant trend the brand should act on

Respond ONLY with valid JSON:
{{"relevance_score": 0.0, "why_relevant": "brief explanation in one sentence"}}"""


@celery_app.task(name="llm_score.llm_score_batch", bind=True, max_retries=2)
def llm_score_batch(self, brand_id: str, post_ids: list[str]) -> dict:
    """
    Score up to LLM_CANDIDATES_PER_CYCLE posts for a brand.
    Sorts by classifier score descending, takes top N.
    """
    MAX_CANDIDATES = settings.llm_candidates_per_cycle  # 20

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # Get brand profile
            cur.execute(
                "SELECT profile_json FROM brands WHERE id = %s",
                (uuid.UUID(brand_id),),
            )
            brand_row = cur.fetchone()
            if not brand_row:
                return {"scored": 0, "error": "brand not found"}
            profile = brand_row[0]

            # Get post texts, ordered by engagement (proxy for classifier score)
            cur.execute(
                """
                SELECT id, text, engagement_count
                FROM posts
                WHERE id = ANY(%s)
                ORDER BY engagement_count DESC
                LIMIT %s
                """,
                (post_ids, MAX_CANDIDATES),
            )
            posts = cur.fetchall()
    finally:
        conn.close()

    template_text, template_id = _get_active_prompt(brand_id)
    scored_posts = []

    for post_id, post_text, engagement in posts:
        prompt = template_text.format(
            brand_name=profile.get("brand_name", ""),
            category=profile.get("category", ""),
            target_audience=profile.get("target_audience", ""),
            core_topics=", ".join(profile.get("core_topics", [])),
            post_text=post_text,
        )

        try:
            message = anthropic_client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(message.content[0].text)
            relevance_score = float(result.get("relevance_score", 0))
            why_relevant = result.get("why_relevant", "")
        except Exception as e:
            # Skip this post on LLM error — don't block the pipeline
            continue

        if relevance_score >= settings.llm_score_threshold:
            scored_posts.append({
                "post_id": post_id,
                "relevance_score": relevance_score,
                "why_relevant": why_relevant,
                "template_id": template_id,
            })

    if scored_posts:
        from services.worker.tasks.assemble import assemble_alerts
        assemble_alerts.delay(brand_id, scored_posts)

    return {"scored": len(posts), "passed_threshold": len(scored_posts)}
```

## services/worker/tasks/assemble.py

```python
"""
Alert Assembler:
1. Runs HDBSCAN on post embeddings to find trend clusters
2. Computes composite score per post
3. Enforces max 10 alerts/brand/day
4. Upserts alerts to Postgres (idempotent)
"""
import uuid
import math
import psycopg2
import numpy as np
import hdbscan
from datetime import datetime, date
from sqlalchemy.orm import Session

from services.worker.celery_app import celery_app
from shared.models import Alert
from shared.settings import Settings
from shared.db import sync_engine

settings = Settings()


def _get_conn():
    return psycopg2.connect(settings.database_sync_url)


def _get_velocity_score(post_id: str) -> float:
    import redis
    r = redis.from_url(settings.redis_url)
    val = r.get(f"velocity:{post_id}")
    return float(val) if val else 1.0


def _compute_novelty(post_id: str, brand_id: str, conn) -> float:
    """
    Novelty = 1 - max cosine similarity to posts that already have alerts.
    High novelty means this post covers new ground.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.embedding
            FROM posts p
            JOIN alerts a ON a.post_id = p.id
            WHERE a.brand_id = %s
              AND a.created_at > NOW() - INTERVAL '7 days'
              AND p.embedding IS NOT NULL
            LIMIT 100
            """,
            (uuid.UUID(brand_id),),
        )
        existing_embeddings = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT embedding FROM posts WHERE id = %s", (post_id,))
        row = cur.fetchone()
        if not row or not row[0] or not existing_embeddings:
            return 1.0

        # Parse embeddings
        def parse_vec(v):
            return np.array([float(x) for x in v.strip("[]").split(",")])

        target = parse_vec(row[0])
        max_similarity = max(
            1 - float(np.dot(target, parse_vec(e)) / (np.linalg.norm(target) * np.linalg.norm(parse_vec(e)) + 1e-9))
            for e in existing_embeddings
        )
        return min(max_similarity, 1.0)


@celery_app.task(name="assemble.assemble_alerts", bind=True, max_retries=3)
def assemble_alerts(self, brand_id: str, scored_posts: list[dict]) -> dict:
    """
    Assemble final alerts from LLM-scored posts.
    scored_posts: [{"post_id": str, "relevance_score": float, "why_relevant": str, "template_id": str|None}]
    """
    MAX_ALERTS_PER_DAY = settings.max_alerts_per_brand_per_day  # 10

    if not scored_posts:
        return {"alerts_created": 0}

    conn = _get_conn()
    try:
        # Check how many alerts already today
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM alerts
                WHERE brand_id = %s
                  AND created_at::date = CURRENT_DATE
                """,
                (uuid.UUID(brand_id),),
            )
            alerts_today = cur.fetchone()[0]

        remaining_slots = MAX_ALERTS_PER_DAY - alerts_today
        if remaining_slots <= 0:
            return {"alerts_created": 0, "reason": "daily limit reached"}

        # Get post embeddings for HDBSCAN clustering
        post_ids = [p["post_id"] for p in scored_posts]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding, engagement_count FROM posts WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (post_ids,),
            )
            rows = cur.fetchall()

        if not rows:
            return {"alerts_created": 0}

        ids_with_emb = [r[0] for r in rows]
        engagements = {r[0]: r[2] for r in rows}

        def parse_vec(v):
            return np.array([float(x) for x in v.strip("[]").split(",")])

        embeddings_matrix = np.array([parse_vec(r[1]) for r in rows])

        # HDBSCAN clustering — no fixed cluster count, noise posts get label -1
        if len(embeddings_matrix) >= 3:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
            labels = clusterer.fit_predict(embeddings_matrix)
        else:
            labels = np.array([-1] * len(embeddings_matrix))

        cluster_map = dict(zip(ids_with_emb, labels.tolist()))

        # Compute composite scores
        max_engagement = max(engagements.values()) or 1

        post_scores = {}
        for sp in scored_posts:
            pid = sp["post_id"]
            if pid not in cluster_map:
                continue
            velocity = _get_velocity_score(pid)
            novelty = _compute_novelty(pid, brand_id, conn)
            eng = engagements.get(pid, 0)

            composite = (
                0.35 * sp["relevance_score"]
                + 0.35 * min(velocity / 5.0, 1.0)  # normalize velocity to 0-1
                + 0.20 * (math.log(eng + 1) / math.log(max_engagement + 1))
                + 0.10 * novelty
            )
            post_scores[pid] = {
                "composite": composite,
                "relevance": sp["relevance_score"],
                "velocity": velocity,
                "engagement": math.log(eng + 1) / math.log(max_engagement + 1),
                "novelty": novelty,
                "why_relevant": sp["why_relevant"],
                "cluster_id": cluster_map[pid],
                "template_id": sp.get("template_id"),
            }

        # Sort by composite score, take top N slots
        ranked = sorted(post_scores.items(), key=lambda x: x[1]["composite"], reverse=True)
        to_create = ranked[:remaining_slots]

        # Upsert alerts (idempotent via unique constraint on brand_id + post_id)
        created = 0
        with conn.cursor() as cur:
            for post_id, scores in to_create:
                cur.execute(
                    """
                    INSERT INTO alerts
                        (id, brand_id, post_id, cluster_id, relevance_score, velocity_score,
                         engagement_score, novelty_score, composite_score, why_relevant, prompt_template_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (brand_id, post_id) DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        uuid.UUID(brand_id),
                        post_id,
                        scores["cluster_id"] if scores["cluster_id"] != -1 else None,
                        scores["relevance"],
                        scores["velocity"],
                        scores["engagement"],
                        scores["novelty"],
                        scores["composite"],
                        scores["why_relevant"],
                        scores["template_id"],
                    ),
                )
                created += 1

        conn.commit()
    finally:
        conn.close()

    return {"alerts_created": created, "brand_id": brand_id}
```
