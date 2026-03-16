# Seed Script + Demo Runner

## scripts/seed_brands.py

```python
"""
Seed 3 sample brands with profiles and query terms.
Run: docker compose run --rm worker python scripts/seed_brands.py
"""
import uuid
from sqlalchemy.orm import Session
from shared.models import Brand, QueryTerm
from shared.db import sync_engine

BRANDS = [
    {
        "name": "PeakHydrate",
        "profile": {
            "brand_name": "PeakHydrate",
            "category": "beverages",
            "target_audience": "health-conscious millennials aged 25-35",
            "core_topics": ["hydration", "wellness", "morning routine", "electrolytes", "fitness recovery"],
            "tone_of_voice": "energetic, science-backed",
            "off_limits_topics": ["alcohol", "unhealthy snacking", "sedentary lifestyle"],
            "relevant_hashtags": ["hydration", "wellness", "morningroutine", "fitness"],
            "competitor_brands": ["GatoradeX", "HydroBoost"],
        },
        "core_terms": ["hydration", "wellness", "morning routine", "electrolytes", "fitness recovery"],
        "exclusions": ["alcohol", "junk food"],
    },
    {
        "name": "CrunchSnacks",
        "profile": {
            "brand_name": "CrunchSnacks",
            "category": "snacks",
            "target_audience": "on-the-go professionals aged 22-40",
            "core_topics": ["healthy snacking", "productivity", "work from home", "afternoon energy", "clean ingredients"],
            "tone_of_voice": "playful, straightforward",
            "off_limits_topics": ["meal replacement", "weight loss claims"],
            "relevant_hashtags": ["healthysnacking", "productivity", "snacktime", "cleaneat"],
            "competitor_brands": ["ProSnack", "NatureBite"],
        },
        "core_terms": ["healthy snacking", "productivity", "work from home", "afternoon energy", "clean ingredients"],
        "exclusions": ["meal replacement", "weight loss"],
    },
    {
        "name": "VitalBoost",
        "profile": {
            "brand_name": "VitalBoost",
            "category": "energy drinks",
            "target_audience": "active young adults aged 18-30",
            "core_topics": ["energy", "focus", "gaming", "esports", "deep work", "pre-workout"],
            "tone_of_voice": "bold, performance-driven",
            "off_limits_topics": ["alcohol mixing", "extreme sports injuries"],
            "relevant_hashtags": ["energy", "focus", "gaming", "deepwork", "preworkout"],
            "competitor_brands": ["EnergyX", "FocusDrink"],
        },
        "core_terms": ["energy", "focus", "gaming", "deep work", "pre-workout"],
        "exclusions": ["alcohol", "extreme sports injuries"],
    },
]


def seed() -> None:
    with Session(sync_engine) as session:
        for brand_data in BRANDS:
            # Idempotent: skip if exists
            existing = session.query(Brand).filter_by(name=brand_data["name"]).first()
            if existing:
                print(f"Brand {brand_data['name']} already exists, skipping")
                continue

            brand = Brand(
                id=uuid.uuid4(),
                name=brand_data["name"],
                profile_json=brand_data["profile"],
            )
            session.add(brand)
            session.flush()

            # Add core terms
            for term in brand_data["core_terms"]:
                session.add(QueryTerm(
                    brand_id=brand.id,
                    term=term,
                    term_type="core",
                    weight=1.0,
                    source="manual",
                    version=1,
                ))

            # Add exclusion terms
            for term in brand_data["exclusions"]:
                session.add(QueryTerm(
                    brand_id=brand.id,
                    term=term,
                    term_type="exclusion",
                    weight=1.0,
                    source="manual",
                    version=1,
                ))

            print(f"Seeded brand: {brand_data['name']} (id={brand.id})")

        session.commit()
    print("Seeding complete.")


if __name__ == "__main__":
    seed()
```

## scripts/run_demo.py

```python
"""
End-to-end demo: triggers ingest + processing for all brands,
then prints the resulting alerts.

Run: docker compose run --rm worker python scripts/run_demo.py
"""
import time
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from shared.models import Brand, Alert, Post
from shared.db import sync_engine
from services.worker.tasks.ingest import ingest_posts_for_brand


def run_demo() -> None:
    with Session(sync_engine) as session:
        brands = session.query(Brand).all()

    if not brands:
        print("No brands seeded. Run: make seed")
        return

    print(f"Triggering ingest for {len(brands)} brands...")
    for brand in brands:
        result = ingest_posts_for_brand(str(brand.id))  # synchronous for demo
        print(f"  {brand.name}: {result}")

    print("\nWaiting 10s for pipeline to complete...")
    time.sleep(10)

    with Session(sync_engine) as session:
        rows = session.execute(
            select(Alert, Post.text, Post.platform, Brand.name)
            .join(Post, Post.id == Alert.post_id)
            .join(Brand, Brand.id == Alert.brand_id)
            .order_by(desc(Alert.composite_score))
            .limit(20)
        ).all()

    print(f"\n{'='*60}")
    print(f"TOP ALERTS ({len(rows)} total)")
    print(f"{'='*60}")
    for alert, post_text, platform, brand_name in rows:
        print(f"\n[{brand_name}] Score: {alert.composite_score:.3f}")
        print(f"Platform: {platform}")
        print(f"Post: {post_text[:100]}...")
        if alert.why_relevant:
            print(f"Why: {alert.why_relevant}")


if __name__ == "__main__":
    run_demo()
```
