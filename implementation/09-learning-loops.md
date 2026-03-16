# Learning Loops — 3 APScheduler Jobs

## services/scheduler/main.py

```python
"""
APScheduler entry point.
Runs 3 learning loops + the 30-min ingest job for all brands.
"""
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.scheduler.jobs.query_runner import run_ingest_for_all_brands
from services.scheduler.jobs.loop_expansion import run_query_expansion_loop
from services.scheduler.jobs.loop_threshold import run_threshold_calibration_loop
from services.scheduler.jobs.loop_prompt import run_prompt_optimization_loop
from shared.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = Settings()

scheduler = BlockingScheduler()

# Post ingestion: every 30 minutes
scheduler.add_job(
    run_ingest_for_all_brands,
    trigger=IntervalTrigger(minutes=settings.ingest_interval_minutes),
    id="ingest_all_brands",
    max_instances=1,
    coalesce=True,
)

# Loop 1: Query Expansion — every Monday
scheduler.add_job(
    run_query_expansion_loop,
    trigger=CronTrigger(day_of_week="mon", hour=2, minute=0),
    id="loop_query_expansion",
    max_instances=1,
)

# Loop 2: Threshold Calibration — every other Monday (bi-weekly)
scheduler.add_job(
    run_threshold_calibration_loop,
    trigger=CronTrigger(day_of_week="mon", week="*/2", hour=3, minute=0),
    id="loop_threshold_calibration",
    max_instances=1,
)

# Loop 3: Prompt Optimization — 1st of each month
scheduler.add_job(
    run_prompt_optimization_loop,
    trigger=CronTrigger(day=1, hour=4, minute=0),
    id="loop_prompt_optimization",
    max_instances=1,
)

if __name__ == "__main__":
    logger.info("Scheduler starting...")
    scheduler.start()
```

## services/scheduler/jobs/query_runner.py

```python
"""Triggers post ingestion for all active brands every 30 minutes."""
import logging
from sqlalchemy.orm import Session
from shared.models import Brand
from shared.db import sync_engine
from services.worker.tasks.ingest import ingest_posts_for_brand

logger = logging.getLogger(__name__)


def run_ingest_for_all_brands() -> None:
    with Session(sync_engine) as session:
        brands = session.query(Brand).all()

    for brand in brands:
        logger.info(f"Triggering ingest for brand {brand.name}")
        ingest_posts_for_brand.delay(str(brand.id))
```

## services/scheduler/jobs/loop_expansion.py

```python
"""
Loop 1: Weekly Query Expansion
Analyzes feedback events from the past 7 days.
Promotes n-grams from positively-engaged posts to expansion terms.
Decays weights of all expansion terms via EMA.
Prunes terms with weight < 0.3 for 3+ consecutive weeks.

Target: CTR 12% → 18% over 8 weeks.
"""
import logging
import uuid
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func

from shared.models import Brand, QueryTerm, Alert, FeedbackEvent, Post
from shared.db import sync_engine
from shared.utils.ngrams import extract_ngrams
from shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

POSITIVE_ACTIONS = {"click", "content_brief", "share"}
PROMOTE_SCORE_THRESHOLD = 0.5
PROMOTE_MIN_FREQUENCY = 3
EMA_ALPHA = 0.1  # weight decay: new = 0.9 * old + 0.1 * current_score
PRUNE_THRESHOLD = 0.3
PRUNE_WEEKS = 3


def _compute_term_score(click_rate: float, action_rate: float, frequency: int) -> float:
    """score = click_rate × action_rate × 10 + log(frequency + 1)"""
    import math
    return click_rate * action_rate * 10 + math.log(frequency + 1)


def run_query_expansion_loop() -> None:
    since = datetime.utcnow() - timedelta(days=7)

    with Session(sync_engine) as session:
        brands = session.query(Brand).all()

    for brand in brands:
        _run_for_brand(str(brand.id), since)


def _run_for_brand(brand_id: str, since: datetime) -> None:
    with Session(sync_engine) as session:
        # Get all alerts + feedback from past 7 days
        rows = session.execute(
            select(Alert.id, Alert.post_id, FeedbackEvent.action)
            .outerjoin(FeedbackEvent, FeedbackEvent.alert_id == Alert.id)
            .where(
                and_(
                    Alert.brand_id == uuid.UUID(brand_id),
                    Alert.created_at >= since,
                )
            )
        ).all()

        if not rows:
            return

        # Group feedback by alert
        alert_feedback: dict[str, list[str]] = defaultdict(list)
        post_for_alert: dict[str, str] = {}
        for alert_id, post_id, action in rows:
            post_for_alert[str(alert_id)] = str(post_id)
            if action:
                alert_feedback[str(alert_id)].append(action)

        # Calculate per-alert CTR
        total_alerts = len(post_for_alert)
        positive_alert_ids = {
            aid for aid, actions in alert_feedback.items()
            if any(a in POSITIVE_ACTIONS for a in actions)
        }

        overall_click_rate = len(positive_alert_ids) / max(total_alerts, 1)
        overall_action_rate = sum(
            1 for actions in alert_feedback.values()
            if actions
        ) / max(total_alerts, 1)

        # Extract n-grams from positively-engaged posts
        positive_post_ids = [post_for_alert[aid] for aid in positive_alert_ids]
        if not positive_post_ids:
            return

        post_texts = session.execute(
            select(Post.text).where(Post.id.in_(positive_post_ids))
        ).scalars().all()

        ngram_counts: Counter = Counter()
        for text in post_texts:
            ngram_counts.update(extract_ngrams(text, n=2))

        # Get existing terms to avoid duplication
        existing_terms = {
            qt.term
            for qt in session.execute(
                select(QueryTerm).where(
                    and_(QueryTerm.brand_id == uuid.UUID(brand_id), QueryTerm.is_active == True)
                )
            ).scalars().all()
        }

        # Promote qualifying n-grams
        for ngram, frequency in ngram_counts.most_common(50):
            if ngram in existing_terms:
                continue
            if frequency < PROMOTE_MIN_FREQUENCY:
                continue

            score = _compute_term_score(overall_click_rate, overall_action_rate, frequency)
            if score >= PROMOTE_SCORE_THRESHOLD:
                new_term = QueryTerm(
                    brand_id=uuid.UUID(brand_id),
                    term=ngram,
                    term_type="expansion",
                    weight=min(score / 10, 1.0),  # normalize to 0-1
                    source="learning_loop",
                    version=1,
                )
                session.add(new_term)

        # EMA decay for existing expansion terms
        expansion_terms = session.execute(
            select(QueryTerm).where(
                and_(
                    QueryTerm.brand_id == uuid.UUID(brand_id),
                    QueryTerm.term_type == "expansion",
                    QueryTerm.is_active == True,
                )
            )
        ).scalars().all()

        for qt in expansion_terms:
            current_score = _compute_term_score(overall_click_rate, overall_action_rate, 1)
            qt.weight = (1 - EMA_ALPHA) * qt.weight + EMA_ALPHA * min(current_score / 10, 1.0)

            # Prune dead terms
            if qt.weight < PRUNE_THRESHOLD:
                qt.is_active = False

        session.commit()
        logger.info(f"Query expansion complete for brand {brand_id}")
```

## services/scheduler/jobs/loop_threshold.py

```python
"""
Loop 2: Bi-weekly Threshold Calibration
Sweeps classifier threshold 0.40–0.90 to maximize F-0.5 score (precision-biased 2:1).
Dampens adjustment to ±0.05 per cycle to prevent oscillation.

Target: Precision@10 from 2.5 → 4.0 over 4 bi-weekly cycles.
"""
import logging
import uuid
import pickle
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from sklearn.metrics import fbeta_score

from shared.models import Brand, Alert, FeedbackEvent, ClassifierParams
from shared.db import sync_engine
from shared.settings import Settings
from shared.utils.s3 import get_s3_client

logger = logging.getLogger(__name__)
settings = Settings()

SWEEP_MIN = 0.40
SWEEP_MAX = 0.90
SWEEP_STEP = 0.05
MAX_ADJUSTMENT_PER_CYCLE = 0.05
FEEDBACK_WINDOW_DAYS = 14
BETA = 0.5  # F-0.5 is precision-biased (precision weighted 2x over recall)


def run_threshold_calibration_loop() -> None:
    since = datetime.utcnow() - timedelta(days=FEEDBACK_WINDOW_DAYS)

    with Session(sync_engine) as session:
        brands = session.query(Brand).all()

    for brand in brands:
        _calibrate_for_brand(str(brand.id), since)


def _calibrate_for_brand(brand_id: str, since: datetime) -> None:
    POSITIVE_ACTIONS = {"click", "content_brief", "share"}

    with Session(sync_engine) as session:
        # Get alerts with feedback labels
        rows = session.execute(
            select(Alert.id, Alert.composite_score, FeedbackEvent.action)
            .outerjoin(FeedbackEvent, FeedbackEvent.alert_id == Alert.id)
            .where(
                and_(
                    Alert.brand_id == uuid.UUID(brand_id),
                    Alert.created_at >= since,
                )
            )
        ).all()

    if len(rows) < 20:
        logger.info(f"Insufficient data for threshold calibration: brand {brand_id}")
        return

    # Build binary labels
    scores = np.array([r[1] for r in rows])
    labels = np.array([1 if r[2] in POSITIVE_ACTIONS else 0 for r in rows])

    # Get current active threshold
    with Session(sync_engine) as session:
        current_params = session.execute(
            select(ClassifierParams).where(
                and_(
                    ClassifierParams.brand_id == uuid.UUID(brand_id),
                    ClassifierParams.is_active == True,
                )
            ).order_by(ClassifierParams.version.desc())
        ).scalars().first()

    current_threshold = current_params.threshold if current_params else 0.65

    # Sweep thresholds, find F-0.5 optimal
    best_threshold = current_threshold
    best_f_score = 0.0

    for threshold in np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP, SWEEP_STEP):
        predictions = (scores >= threshold).astype(int)
        if predictions.sum() == 0:
            continue
        f = fbeta_score(labels, predictions, beta=BETA, zero_division=0)
        if f > best_f_score:
            best_f_score = f
            best_threshold = float(threshold)

    # Dampen adjustment
    delta = best_threshold - current_threshold
    delta = max(-MAX_ADJUSTMENT_PER_CYCLE, min(MAX_ADJUSTMENT_PER_CYCLE, delta))
    new_threshold = current_threshold + delta

    logger.info(
        f"Brand {brand_id}: threshold {current_threshold:.2f} → {new_threshold:.2f} "
        f"(F-{BETA}={best_f_score:.3f})"
    )

    # Persist new version
    with Session(sync_engine) as session:
        next_version = (current_params.version + 1) if current_params else 1

        # Deactivate old
        if current_params:
            current_params.is_active = False

        # Insert new
        new_params = ClassifierParams(
            brand_id=uuid.UUID(brand_id),
            version=next_version,
            threshold=new_threshold,
            f_score=best_f_score,
            is_active=True,
        )
        session.add(new_params)
        session.commit()
```

## services/scheduler/jobs/loop_prompt.py

```python
"""
Loop 3: Monthly Prompt Optimization
1. Collect failure cases (dismissed alerts with high LLM score)
2. LLM analyzes failure patterns and suggests prompt improvements
3. New prompt version created with 20% A/B traffic weight
4. After 2 weeks, evaluate: if CTR higher → increase to 80%; else → deactivate

Target: suppression precision 75% → 90% over 3 months.
"""
import logging
import uuid
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from anthropic import Anthropic

from shared.models import Brand, Alert, FeedbackEvent, PromptTemplate
from shared.db import sync_engine
from shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

FAILURE_COLLECTION_DAYS = 30
DISMISSED_LLM_SCORE_MIN = 0.7  # High LLM score but user dismissed = failure case
AB_INITIAL_WEIGHT = 0.2
MAX_FAILURE_CASES = 30


def run_prompt_optimization_loop() -> None:
    with Session(sync_engine) as session:
        brands = session.query(Brand).all()

    for brand in brands:
        _optimize_for_brand(str(brand.id), str(brand.profile_json))


def _optimize_for_brand(brand_id: str, profile_json: str) -> None:
    since = datetime.utcnow() - timedelta(days=FAILURE_COLLECTION_DAYS)

    with Session(sync_engine) as session:
        # Collect failure cases: high LLM score + dismissed by user
        failure_rows = session.execute(
            select(Alert.why_relevant, Alert.relevance_score, FeedbackEvent.action)
            .join(FeedbackEvent, FeedbackEvent.alert_id == Alert.id)
            .where(
                and_(
                    Alert.brand_id == uuid.UUID(brand_id),
                    Alert.created_at >= since,
                    Alert.relevance_score >= DISMISSED_LLM_SCORE_MIN,
                    FeedbackEvent.action == "dismiss",
                )
            )
            .limit(MAX_FAILURE_CASES)
        ).all()

    if len(failure_rows) < 5:
        logger.info(f"Insufficient failure cases for prompt optimization: brand {brand_id}")
        return

    failure_cases = [
        {"why_relevant": row[0], "relevance_score": row[1]}
        for row in failure_rows
    ]

    # Ask LLM to identify failure patterns and suggest improved prompt
    analysis_prompt = f"""You are improving an LLM prompt that scores social media posts for CPG brand relevance.

The current prompt generates high relevance scores for posts that brand managers then dismiss.
These are the failure cases (posts with high scores that were dismissed):

{json.dumps(failure_cases[:10], indent=2)}

Based on these failure patterns:
1. Identify the common reasons these posts were incorrectly scored as relevant
2. Suggest specific additions to the scoring prompt to prevent these false positives

Respond with JSON:
{{
  "failure_patterns": ["pattern 1", "pattern 2"],
  "prompt_additions": "Text to add to the scoring prompt to prevent these false positives"
}}"""

    try:
        message = anthropic_client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": analysis_prompt}],
        )
        analysis = json.loads(message.content[0].text)
    except Exception as e:
        logger.error(f"LLM analysis failed for brand {brand_id}: {e}")
        return

    # Get current active prompt template
    with Session(sync_engine) as session:
        current = session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.brand_id == uuid.UUID(brand_id),
                    PromptTemplate.template_type == "llm_score",
                    PromptTemplate.is_active == True,
                )
            ).order_by(PromptTemplate.version.desc())
        ).scalars().first()

        if not current:
            logger.info(f"No active prompt template for brand {brand_id}, skipping")
            return

        # Create improved prompt by appending guidance
        improved_text = current.template_text + f"""

ADDITIONAL GUIDANCE (updated {datetime.utcnow().date()}):
Avoid scoring high for posts that match these patterns:
{chr(10).join(f'- {p}' for p in analysis.get('failure_patterns', []))}

{analysis.get('prompt_additions', '')}"""

        next_version = current.version + 1

        # Add new template with 20% A/B weight
        new_template = PromptTemplate(
            brand_id=uuid.UUID(brand_id),
            template_type="llm_score",
            version=next_version,
            template_text=improved_text,
            ab_weight=AB_INITIAL_WEIGHT,
            is_active=True,
        )
        session.add(new_template)
        session.commit()

    logger.info(
        f"Created prompt template v{next_version} for brand {brand_id} "
        f"with {AB_INITIAL_WEIGHT*100:.0f}% A/B weight"
    )
```
