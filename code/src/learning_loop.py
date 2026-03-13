"""
Query Expansion Learning Loop — End-to-End Demonstration

This module implements the Query Expansion Learning Loop described in the
architecture proposal (Section 4.1). It demonstrates:

  1. Mock feedback dataset generation (realistic click/action signals)
  2. N-gram extraction from positively-engaged posts
  3. Term scoring and weight update procedure
  4. CTR evaluation before and after learning
  5. Overfitting verification on training data (required sanity check)

Design note on overfitting:
  The learning loop is designed to overfit the training data intentionally.
  It extracts n-grams from positively-engaged posts, scores them by their
  click/action rates within the training window, and uses those scores to
  preferentially surface similar posts in the future. "Overfitting" here
  means: if we replay the same posts through the updated filter, CTR improves.
  Generalization is validated separately via live CTR on new posts.

Usage:
    python3 learning_loop.py

Expected output: CTR improvement on training data, updated term weights per brand.
"""

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Post:
    post_id: str
    brand_id: str
    text: str
    platform: str
    likes: int
    reposts: int


@dataclass
class Alert:
    alert_id: str
    brand_id: str
    post_id: str
    relevance_score: float  # initial classifier score
    created_at: datetime


@dataclass
class FeedbackEvent:
    feedback_id: str
    alert_id: str
    brand_id: str
    post_id: str
    action: str  # 'click' | 'dismiss' | 'content_brief' | 'share'
    dwell_time_ms: int
    created_at: datetime


@dataclass
class QueryTerm:
    brand_id: str
    term: str
    weight: float       # 0.0–1.0; terms below 0.25 are pruned
    term_type: str      # 'core' | 'expansion'
    source: str         # 'guidelines' | 'learning_loop'


# ---------------------------------------------------------------------------
# Brand definitions
# ---------------------------------------------------------------------------

# Each positive_topics entry has 4+ keywords so that template expansion
# produces varied but topic-consistent post text.
BRAND_DEFINITIONS = {
    "zephyr_energy": {
        "core_terms": ["energy drink", "focus", "productivity", "morning routine"],
        # Positive topics: vocabulary brand managers want to engage with.
        # Each entry becomes 3 posts → topic n-grams appear 3 times → pass min_frequency.
        "positive_topics": [
            "afternoon slump deep work flow state cognitive performance",
            "morning routine daily ritual wake up early 5am club",
            "clean energy no crash L-theanine natural caffeine sustained focus",
            "productivity hack time blocking pomodoro technique GTD method",
            "work from home hybrid work focus session deep concentration",
        ],
        # Negative topics: off-brand / off-limits content.
        "negative_topics": [
            "extreme sports dangerous stunts adrenaline rush",
            "all nighter sleep deprivation weight loss diet culture",
        ],
    },
    "brighten_home": {
        "core_terms": ["plant-based cleaner", "non-toxic", "eco friendly", "sustainable"],
        "positive_topics": [
            "pet safe cleaning dog friendly cat safe animal products",
            "apartment cleaning small space studio renter cleaning tips",
            "refill station zero waste reduce plastic sustainable routine",
            "clean with me cleaning routine sunday reset deep clean",
            "transparent ingredients safe for kids family household",
        ],
        "negative_topics": [
            "political environmental activism protest",
            "chemical free natural unsubstantiated health claims",
        ],
    },
    "trailblaze": {
        "core_terms": ["trail snacks", "hiking food", "outdoor snacks", "backpacking"],
        "positive_topics": [
            "solo hiking trail safety women hiking empowerment",
            "beginner hiker first trail easy hikes newbie outdoors",
            "national park best trails trail recommendation weekend hike",
            "hiking with dogs pet friendly trail dog hike",
            "leave no trace trail ethics outdoor community responsible",
        ],
        "negative_topics": [
            "gatekeeping real hikers extreme mountaineering elitism",
            "weight loss earn your calories diet restriction",
        ],
    },
}

PLATFORMS = ["twitter", "reddit", "instagram", "tiktok"]

# Templates that naturally embed the topic phrase in the post text.
# The {topic} placeholder is filled with a 2-3 word phrase from the topic.
POST_TEMPLATES = [
    "just discovered {topic} and it completely changed how I work",
    "anyone else using {topic} daily? sharing what worked for me",
    "day 14 of {topic} — here is what I noticed after two weeks",
    "the {topic} community is surprisingly supportive and helpful",
    "{topic} saved my routine this week — highly recommend trying it",
    "why I finally started taking {topic} seriously after years of ignoring it",
    "tested {topic} for 30 days — honest review with pros and cons",
    "can we talk about how underrated {topic} actually is for real",
    "my full experience with {topic} after three months of daily use",
    "switched to {topic} six months ago and have not looked back since",
    "everything I wish I knew about {topic} before starting out",
    "sharing my honest thoughts on {topic} so you can decide yourself",
]


# ---------------------------------------------------------------------------
# 1. Mock dataset generation
# ---------------------------------------------------------------------------

def _sample_topic_phrase(topic_text: str, rng: random.Random) -> str:
    """Pick a 2-3 word contiguous phrase from a topic string."""
    words = topic_text.split()
    if len(words) < 2:
        return words[0]
    start = rng.randint(0, len(words) - 2)
    length = rng.randint(2, min(3, len(words) - start))
    return " ".join(words[start: start + length])


def _make_post(post_id: str, brand_id: str, topic_text: str, likes: int,
               rng: random.Random) -> Post:
    """Create a single mock post by sampling a template and topic phrase."""
    phrase = _sample_topic_phrase(topic_text, rng)
    template = rng.choice(POST_TEMPLATES)
    text = template.format(topic=phrase)
    platform = rng.choice(PLATFORMS)
    reposts = max(0, int(likes * rng.uniform(0.05, 0.25)))
    return Post(
        post_id=post_id,
        brand_id=brand_id,
        text=text,
        platform=platform,
        likes=likes,
        reposts=reposts,
    )


def generate_mock_dataset(
    seed: int = 42,
    posts_per_positive_topic: int = 3,
    posts_per_negative_topic: int = 2,
    positive_feedback_rate: float = 0.72,
    negative_feedback_rate: float = 0.80,
) -> Tuple[List[Post], List[Alert], List[FeedbackEvent]]:
    """
    Generate a realistic mock feedback dataset.

    Structure per brand:
      - posts_per_positive_topic × 5 positive topics = 15 positive posts
      - posts_per_negative_topic × 2 negative topics = 4  negative posts
    Total per brand: 19 posts; overall: 57 posts across 3 brands.

    Having 3 posts per positive topic ensures that topic-specific n-grams
    reach min_frequency=2, which is required for the learning loop to
    promote them as expansion terms.

    Feedback:
      - Positive posts: ~72% get positive action (click/content_brief/share)
      - Negative posts: ~80% get dismissed
    """
    rng = random.Random(seed)
    base_time = datetime(2025, 1, 15, 12, 0, 0)

    posts: List[Post] = []
    alerts: List[Alert] = []
    feedbacks: List[FeedbackEvent] = []

    alert_counter = 0
    feedback_counter = 0

    for brand_id, brand_def in BRAND_DEFINITIONS.items():
        # --- Positive topic posts ---
        for topic_idx, topic in enumerate(brand_def["positive_topics"]):
            for variant in range(posts_per_positive_topic):
                post_id = f"{brand_id[:3]}_pos_{topic_idx}_{variant}"
                # High engagement: likes 200-2000
                likes = rng.randint(200, 2000)
                post = _make_post(post_id, brand_id, topic, likes, rng)
                posts.append(post)

                alert_id = f"alert_{alert_counter:04d}"
                alert_counter += 1
                alert = Alert(
                    alert_id=alert_id,
                    brand_id=brand_id,
                    post_id=post_id,
                    relevance_score=round(rng.uniform(0.62, 0.95), 3),
                    created_at=base_time + timedelta(hours=rng.uniform(0, 48)),
                )
                alerts.append(alert)

                if rng.random() < positive_feedback_rate:
                    # Weight toward higher-value actions (content_brief > share > click)
                    action = rng.choices(
                        ["click", "content_brief", "share"],
                        weights=[0.5, 0.3, 0.2],
                    )[0]
                    dwell = rng.randint(15_000, 90_000)
                else:
                    action = "dismiss"
                    dwell = rng.randint(500, 3_000)

                feedbacks.append(FeedbackEvent(
                    feedback_id=f"fb_{feedback_counter:05d}",
                    alert_id=alert_id,
                    brand_id=brand_id,
                    post_id=post_id,
                    action=action,
                    dwell_time_ms=dwell,
                    created_at=alert.created_at + timedelta(minutes=rng.uniform(5, 300)),
                ))
                feedback_counter += 1

        # --- Negative topic posts ---
        for topic_idx, topic in enumerate(brand_def["negative_topics"]):
            for variant in range(posts_per_negative_topic):
                post_id = f"{brand_id[:3]}_neg_{topic_idx}_{variant}"
                # Low engagement: likes 5-80
                likes = rng.randint(5, 80)
                post = _make_post(post_id, brand_id, topic, likes, rng)
                posts.append(post)

                alert_id = f"alert_{alert_counter:04d}"
                alert_counter += 1
                alert = Alert(
                    alert_id=alert_id,
                    brand_id=brand_id,
                    post_id=post_id,
                    relevance_score=round(rng.uniform(0.45, 0.70), 3),
                    created_at=base_time + timedelta(hours=rng.uniform(0, 48)),
                )
                alerts.append(alert)

                if rng.random() < negative_feedback_rate:
                    action = "dismiss"
                    dwell = rng.randint(300, 1_500)
                else:
                    action = "click"
                    dwell = rng.randint(2_000, 8_000)

                feedbacks.append(FeedbackEvent(
                    feedback_id=f"fb_{feedback_counter:05d}",
                    alert_id=alert_id,
                    brand_id=brand_id,
                    post_id=post_id,
                    action=action,
                    dwell_time_ms=dwell,
                    created_at=alert.created_at + timedelta(minutes=rng.uniform(5, 300)),
                ))
                feedback_counter += 1

    return posts, alerts, feedbacks


# ---------------------------------------------------------------------------
# 2. Baseline query terms (from brand guidelines — no learning yet)
# ---------------------------------------------------------------------------

def initialize_query_terms() -> Dict[str, List[QueryTerm]]:
    """Return the starting query terms derived from brand guidelines."""
    terms: Dict[str, List[QueryTerm]] = {}
    for brand_id, brand_def in BRAND_DEFINITIONS.items():
        terms[brand_id] = [
            QueryTerm(
                brand_id=brand_id,
                term=t,
                weight=1.0,
                term_type="core",
                source="guidelines",
            )
            for t in brand_def["core_terms"]
        ]
    return terms


# ---------------------------------------------------------------------------
# 3. Evaluation — CTR calculation
# ---------------------------------------------------------------------------

POSITIVE_ACTIONS = {"click", "content_brief", "share"}


def calculate_ctr(
    alerts: List[Alert],
    feedbacks: List[FeedbackEvent],
    brand_id: Optional[str] = None,
) -> float:
    """
    CTR = positive_actions / total_alerts_with_feedback.

    Args:
        alerts: All alerts in the dataset.
        feedbacks: All feedback events.
        brand_id: If provided, filter to this brand only.

    Returns:
        CTR as a float in [0, 1].
    """
    feedback_by_alert = {fb.alert_id: fb for fb in feedbacks}

    scored = [
        a for a in alerts
        if a.alert_id in feedback_by_alert
        and (brand_id is None or a.brand_id == brand_id)
    ]
    if not scored:
        return 0.0

    positive = sum(
        1 for a in scored
        if feedback_by_alert[a.alert_id].action in POSITIVE_ACTIONS
    )
    return positive / len(scored)


# ---------------------------------------------------------------------------
# 4. Query Expansion Learning Loop
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "my", "me", "i", "it", "its", "this", "that", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "just", "so",
    "not", "here", "after", "from", "what", "how", "why", "when", "who",
    "your", "their", "our", "we", "they", "he", "she", "you", "up", "out",
    "about", "like", "as", "by", "into", "finally", "below", "ever", "even",
    "very", "really", "only", "also", "well", "can", "all", "its",
}


def _extract_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> List[str]:
    """
    Extract n-grams (1–3 words) from lowercased text.
    Removes stopwords and tokens shorter than 3 characters.
    """
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i: i + n]))
    return ngrams


def run_query_expansion_loop(
    posts: List[Post],
    alerts: List[Alert],
    feedbacks: List[FeedbackEvent],
    current_terms: Dict[str, List[QueryTerm]],
    min_frequency: int = 2,
    min_score: float = 0.4,
    weight_decay: float = 0.9,
    max_new_terms_per_brand: int = 5,
) -> Dict[str, List[QueryTerm]]:
    """
    Core Query Expansion Learning Loop.

    For each brand:
      1. Identify positively-engaged posts (click / content_brief / share).
      2. Extract n-grams from those posts; skip grams already in core terms.
      3. For each candidate n-gram, count how many surfaced posts contained it,
         how many were clicked, and how many triggered a high-value action.
      4. Score: score = click_rate * action_rate * 10 + log(frequency + 1)
      5. Promote candidates with score >= min_score and frequency >= min_frequency
         as new expansion terms with weight = min(score / 5.0, 0.95).
      6. Decay existing expansion term weights by weight_decay per cycle.
      7. Prune terms with weight < 0.25.

    Args:
        posts: All posts in the dataset.
        alerts: All alerts (one per post in this demo).
        feedbacks: All feedback events.
        current_terms: Current query term weights per brand.
        min_frequency: Min n-gram occurrences in surfaced posts.
        min_score: Min score threshold for new term promotion.
        weight_decay: Multiplicative decay for existing expansion weights.
        max_new_terms_per_brand: Max new terms added per brand per cycle.

    Returns:
        Updated query terms dict (new object — does not mutate input).
    """
    post_by_id = {p.post_id: p for p in posts}
    feedback_by_alert = {fb.alert_id: fb for fb in feedbacks}

    updated_terms: Dict[str, List[QueryTerm]] = {}

    for brand_id in BRAND_DEFINITIONS:
        existing = current_terms.get(brand_id, [])

        # Build set of n-grams already covered by existing core terms
        existing_core_grams: set = set()
        for qt in existing:
            existing_core_grams.update(_extract_ngrams(qt.term))

        # Collect (post, action) pairs for this brand
        brand_post_actions: List[Tuple[Post, str]] = []
        for alert in alerts:
            if alert.brand_id != brand_id:
                continue
            fb = feedback_by_alert.get(alert.alert_id)
            if fb is None:
                continue
            post = post_by_id.get(alert.post_id)
            if post is None:
                continue
            brand_post_actions.append((post, fb.action))

        if not brand_post_actions:
            updated_terms[brand_id] = list(existing)
            continue

        # Count n-gram occurrences across surfaced / clicked / actioned posts
        ngram_surfaced: Dict[str, int] = defaultdict(int)
        ngram_clicked: Dict[str, int] = defaultdict(int)
        ngram_actioned: Dict[str, int] = defaultdict(int)

        for post, action in brand_post_actions:
            grams = set(_extract_ngrams(post.text))
            for gram in grams:
                if gram in existing_core_grams:
                    continue  # already covered — not a new signal
                ngram_surfaced[gram] += 1
                if action in POSITIVE_ACTIONS:
                    ngram_clicked[gram] += 1
                if action in {"content_brief", "share"}:
                    ngram_actioned[gram] += 1

        # Score each candidate
        candidates: List[Tuple[str, float]] = []
        for gram, freq in ngram_surfaced.items():
            if freq < min_frequency:
                continue
            click_rate = ngram_clicked[gram] / freq
            action_rate = (
                ngram_actioned[gram] / ngram_clicked[gram]
                if ngram_clicked[gram] > 0 else 0.0
            )
            score = click_rate * action_rate * 10 + math.log(freq + 1)
            if score >= min_score:
                candidates.append((gram, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:max_new_terms_per_brand]

        # Rebuild term list
        new_terms: List[QueryTerm] = []

        # Keep core terms unchanged
        for qt in existing:
            if qt.term_type == "core":
                new_terms.append(qt)
            else:
                # Decay expansion terms; prune if weight falls below threshold
                decayed_weight = round(qt.weight * weight_decay, 4)
                if decayed_weight >= 0.25:
                    new_terms.append(QueryTerm(
                        brand_id=qt.brand_id,
                        term=qt.term,
                        weight=decayed_weight,
                        term_type=qt.term_type,
                        source=qt.source,
                    ))

        # Add new expansion terms
        existing_term_set = {qt.term for qt in new_terms}
        for gram, score in top_candidates:
            if gram in existing_term_set:
                continue
            initial_weight = min(round(score / 5.0, 4), 0.95)
            new_terms.append(QueryTerm(
                brand_id=brand_id,
                term=gram,
                weight=initial_weight,
                term_type="expansion",
                source="learning_loop",
            ))
            existing_term_set.add(gram)

        updated_terms[brand_id] = new_terms

    return updated_terms


# ---------------------------------------------------------------------------
# 5. Simulate updated CTR — shows the overfitting effect
# ---------------------------------------------------------------------------

def _post_learned_score(post: Post, terms: List[QueryTerm]) -> float:
    """
    Compute a post's learned relevance score against the updated term list.

    Score = sum of weights of all terms (core + expansion) that appear in
    the post text, capped at 1.0.

    Key property: adding a new expansion term can only INCREASE a post's
    score (never decrease it), because we sum matched weights rather than
    dividing by total weights. This correctly models how the learning loop
    makes the filter more precise — posts containing the learned vocabulary
    score strictly higher.
    """
    text_lower = post.text.lower()
    matched_weight = sum(
        qt.weight for qt in terms if qt.term.lower() in text_lower
    )
    return min(matched_weight, 1.0)


def simulate_updated_ctr(
    posts: List[Post],
    alerts: List[Alert],
    feedbacks: List[FeedbackEvent],
    updated_terms: Dict[str, List[QueryTerm]],
    score_threshold: float = 0.55,
) -> Tuple[float, float]:
    """
    Simulate the CTR achieved by filtering alerts using updated term weights.

    For each alert:
      - Compute the post's learned score against the updated term list.
      - Surface the alert only if learned_score >= score_threshold.
    Compute CTR on the surfaced set.

    A higher CTR (compared to the unfiltered baseline) demonstrates that
    the learned terms preferentially surface posts that brand managers
    engage with — i.e., the loop overfits the training feedback signal.

    Returns:
        (new_ctr, fraction_of_alerts_surfaced)
    """
    post_by_id = {p.post_id: p for p in posts}
    feedback_by_alert = {fb.alert_id: fb for fb in feedbacks}

    total_surfaced = 0
    positive_surfaced = 0

    for alert in alerts:
        post = post_by_id.get(alert.post_id)
        fb = feedback_by_alert.get(alert.alert_id)
        if post is None or fb is None:
            continue

        terms = updated_terms.get(alert.brand_id, [])
        learned_score = _post_learned_score(post, terms)

        if learned_score >= score_threshold:
            total_surfaced += 1
            if fb.action in POSITIVE_ACTIONS:
                positive_surfaced += 1

    if total_surfaced == 0:
        return 0.0, 0.0

    new_ctr = positive_surfaced / total_surfaced
    fraction_surfaced = total_surfaced / max(len(alerts), 1)
    return new_ctr, fraction_surfaced


# ---------------------------------------------------------------------------
# 6. Main — run full loop and print results
# ---------------------------------------------------------------------------

def _format_terms_table(brand_id: str, terms: List[QueryTerm]) -> str:
    lines = [f"\n  Brand: {brand_id}"]
    core = [qt for qt in terms if qt.term_type == "core"]
    expansion = sorted(
        [qt for qt in terms if qt.term_type == "expansion"],
        key=lambda x: x.weight,
        reverse=True,
    )
    lines.append(f"  {'TERM':<38} {'WEIGHT':>6}  {'TYPE':<10}  SOURCE")
    lines.append("  " + "-" * 72)
    for qt in core + expansion:
        lines.append(
            f"  {qt.term:<38} {qt.weight:>6.3f}  {qt.term_type:<10}  {qt.source}"
        )
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  Query Expansion Learning Loop — Demonstration")
    print("=" * 72)

    # ---- Step 1: Generate mock data ----------------------------------------
    print("\n[1] Generating mock dataset...")
    posts, alerts, feedbacks = generate_mock_dataset(seed=42)
    positive_fb = sum(1 for fb in feedbacks if fb.action in POSITIVE_ACTIONS)
    print(f"    Posts:              {len(posts)}")
    print(f"    Alerts:             {len(alerts)}")
    print(f"    Feedbacks:          {len(feedbacks)}")
    print(f"    Positive feedback:  {positive_fb} / {len(feedbacks)}"
          f"  ({positive_fb/len(feedbacks):.0%})")

    # ---- Step 2: Baseline query terms from guidelines ----------------------
    print("\n[2] Initializing baseline query terms from brand guidelines...")
    baseline_terms = initialize_query_terms()
    for brand_id, terms in baseline_terms.items():
        print(f"    {brand_id}: {len(terms)} core terms — "
              + ", ".join(f'"{qt.term}"' for qt in terms))

    # ---- Step 3: CTR BEFORE learning ---------------------------------------
    print("\n[3] CTR BEFORE learning loop (unfiltered — all alerts surfaced):")
    ctr_before = calculate_ctr(alerts, feedbacks)
    print(f"    Overall:  {ctr_before:.1%}")
    for brand_id in BRAND_DEFINITIONS:
        ctr = calculate_ctr(alerts, feedbacks, brand_id=brand_id)
        n_alerts = sum(1 for a in alerts if a.brand_id == brand_id)
        print(f"      {brand_id:<22} {ctr:.1%}  ({n_alerts} alerts)")

    # ---- Step 4: Run learning loop -----------------------------------------
    print("\n[4] Running Query Expansion Learning Loop...")
    updated_terms = run_query_expansion_loop(
        posts=posts,
        alerts=alerts,
        feedbacks=feedbacks,
        current_terms=baseline_terms,
        min_frequency=2,
        min_score=0.4,
        weight_decay=0.9,
        max_new_terms_per_brand=5,
    )
    for brand_id in BRAND_DEFINITIONS:
        before = len(baseline_terms.get(brand_id, []))
        after = len(updated_terms.get(brand_id, []))
        new = after - before
        expansion_terms = [
            qt for qt in updated_terms[brand_id] if qt.term_type == "expansion"
        ]
        top = sorted(expansion_terms, key=lambda x: x.weight, reverse=True)[:3]
        top_str = ", ".join(f'"{qt.term}" ({qt.weight:.2f})' for qt in top)
        print(f"    {brand_id}: +{new} new terms  |  top: {top_str}")

    # ---- Step 5: CTR AFTER learning ----------------------------------------
    print("\n[5] CTR AFTER learning loop (filtering by learned term scores):")
    ctr_after, fraction_surfaced = simulate_updated_ctr(
        posts=posts,
        alerts=alerts,
        feedbacks=feedbacks,
        updated_terms=updated_terms,
        score_threshold=0.55,
    )
    print(f"    Overall:              {ctr_after:.1%}")
    print(f"    Fraction surfaced:    {fraction_surfaced:.1%}  "
          "(updated terms filter out low-relevance posts)")

    improvement_abs = ctr_after - ctr_before
    improvement_rel = (ctr_after / ctr_before - 1) if ctr_before > 0 else 0.0
    print(f"\n    CTR improvement:  {ctr_before:.1%}  →  {ctr_after:.1%}")
    print(f"    Absolute:  +{improvement_abs:.1%}")
    print(f"    Relative:  +{improvement_rel:.0%}")

    if ctr_after > ctr_before:
        print("    ✓ Overfitting confirmed: learning loop improves CTR on training data.")
    else:
        print("    ✗ WARNING: no CTR improvement observed — check learning parameters.")

    # ---- Step 6: Updated term weights (output format) ----------------------
    print("\n[6] Updated query term weights (output format):")
    for brand_id in BRAND_DEFINITIONS:
        print(_format_terms_table(brand_id, updated_terms[brand_id]))

    # ---- Summary -----------------------------------------------------------
    total_new = sum(
        len(updated_terms[b]) - len(baseline_terms[b]) for b in BRAND_DEFINITIONS
    )
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  CTR before learning:          {ctr_before:.1%}")
    print(f"  CTR after learning:           {ctr_after:.1%}  (on training data)")
    print(f"  Total new expansion terms:    {total_new} across {len(BRAND_DEFINITIONS)} brands")
    print("""
  Interpretation:
    The learning loop extracts n-grams from posts that brand managers clicked
    or acted on. These terms are assigned weights proportional to their
    click-rate × action-rate score. When we replay training data through the
    updated filter (score_threshold=0.55), positively-engaged posts score
    higher because they contain the vocabulary the loop just learned — this
    is the intended overfitting effect.

    In production, this overfitting is managed by:
    (a) applying learned terms to NEW posts (not training replays),
    (b) weekly weight decay (×0.9/cycle) preventing stale term accumulation,
    (c) monitoring live CTR as the ground-truth generalization metric.
""")


if __name__ == "__main__":
    main()
