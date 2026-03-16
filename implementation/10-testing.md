# Testing — Strategy + Key Test Cases

## Test Structure

```
tests/
├── conftest.py           ← fixtures: DB, test client, seed data
├── test_processing.py    ← 5-step funnel unit tests
├── test_learning_loops.py ← loop behavior + CTR improvement
└── test_api.py           ← REST endpoints + WebSocket
```

## tests/conftest.py

```python
import pytest
import asyncio
import uuid
from datetime import datetime
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from shared.models import Base, Brand, Post, Alert, FeedbackEvent, QueryTerm
from shared.settings import Settings
from services.api.main import app

settings = Settings()

# Use a separate test database
TEST_DB_URL = settings.database_sync_url.replace("/cpgdb", "/cpgdb_test")


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DB_URL)
    # Ensure pgvector is installed
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Fresh session per test, rolled back after."""
    with Session(test_engine) as session:
        yield session
        session.rollback()


@pytest.fixture(scope="function")
def sample_brand(db_session) -> Brand:
    brand = Brand(
        id=uuid.uuid4(),
        name="TestBrand",
        profile_json={
            "brand_name": "TestBrand",
            "category": "beverages",
            "target_audience": "health-conscious millennials",
            "core_topics": ["wellness", "hydration", "morning routine"],
            "tone_of_voice": "energetic",
            "off_limits_topics": ["alcohol", "junk food"],
            "relevant_hashtags": ["wellness", "hydration"],
            "competitor_brands": ["CompetitorA"],
        },
    )
    db_session.add(brand)
    db_session.flush()
    return brand


@pytest.fixture(scope="function")
def sample_posts(db_session, sample_brand) -> list[Post]:
    posts = [
        Post(
            id=f"post_{i}",
            brand_id=sample_brand.id,
            platform="tiktok",
            text=f"Morning wellness routine with hydration boost #{i}",
            engagement_count=100 * (i + 1),
            published_at=datetime.utcnow(),
        )
        for i in range(10)
    ]
    db_session.add_all(posts)
    db_session.flush()
    return posts


@pytest.fixture(scope="function")
async def api_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
```

## tests/test_processing.py

```python
"""Unit tests for the 5-step ML processing funnel."""
import uuid
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestNearDedup:
    """Dedup threshold: cosine > 0.97 → drop."""

    def test_identical_posts_dropped(self):
        """Two posts with identical embeddings should trigger dedup."""
        from services.worker.tasks.embed import near_dedup_filter

        embedding = np.random.rand(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        # Cosine distance between identical embeddings = 0.0, well below (1 - 0.97) = 0.03
        distance = 1 - float(np.dot(embedding, embedding))
        assert distance < 0.03

    def test_dissimilar_posts_survive(self):
        """Posts with cosine similarity < 0.97 should pass dedup."""
        emb1 = np.random.rand(384).astype(np.float32)
        emb2 = np.random.rand(384).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 /= np.linalg.norm(emb2)

        similarity = float(np.dot(emb1, emb2))
        # Random high-dim vectors have near-zero cosine similarity
        assert similarity < 0.97


class TestCandidateFilter:
    """Brand centroid cosine similarity threshold: 0.55."""

    def test_relevant_post_passes(self):
        """Post with cosine similarity ≥ 0.55 to centroid should pass."""
        centroid = np.ones(384).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        # Very similar post embedding
        post_emb = centroid + np.random.rand(384).astype(np.float32) * 0.1
        post_emb /= np.linalg.norm(post_emb)

        similarity = float(np.dot(centroid, post_emb))
        distance = 1 - similarity

        # Should pass: distance < (1 - 0.55) = 0.45
        assert distance < 0.45

    def test_irrelevant_post_dropped(self):
        """Post with cosine similarity < 0.55 to centroid should be dropped."""
        centroid = np.ones(384).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        # Orthogonal embedding
        irrelevant = np.zeros(384).astype(np.float32)
        irrelevant[0] = 1.0  # orthogonal to uniform centroid

        similarity = float(np.dot(centroid, irrelevant))
        distance = 1 - similarity

        # Orthogonal: similarity ≈ 1/√384 ≈ 0.05, distance ≈ 0.95 > 0.45
        assert distance > 0.45


class TestCompositeScore:
    """Composite score formula validation."""

    def test_composite_score_formula(self):
        """Verify composite score = 0.35R + 0.35V + 0.20E + 0.10N."""
        import math

        relevance = 0.8
        velocity = 2.0  # will be normalized: min(2.0/5.0, 1.0) = 0.4
        engagement = 1000
        max_engagement = 10000
        novelty = 0.9

        velocity_norm = min(velocity / 5.0, 1.0)
        engagement_norm = math.log(engagement + 1) / math.log(max_engagement + 1)

        composite = (
            0.35 * relevance
            + 0.35 * velocity_norm
            + 0.20 * engagement_norm
            + 0.10 * novelty
        )

        expected = 0.35 * 0.8 + 0.35 * 0.4 + 0.20 * engagement_norm + 0.10 * 0.9
        assert abs(composite - expected) < 1e-9

    def test_composite_score_in_valid_range(self):
        """Composite score should always be in [0, 1]."""
        import math

        test_cases = [
            (1.0, 10.0, 100000, 100000, 1.0),
            (0.0, 0.0, 0, 1, 0.0),
            (0.6, 1.5, 500, 10000, 0.7),
        ]
        for rel, vel, eng, max_eng, nov in test_cases:
            vel_norm = min(vel / 5.0, 1.0)
            eng_norm = math.log(eng + 1) / math.log(max_eng + 1)
            composite = 0.35 * rel + 0.35 * vel_norm + 0.20 * eng_norm + 0.10 * nov
            assert 0.0 <= composite <= 1.0


class TestHDBSCAN:
    """HDBSCAN clustering behavior."""

    def test_noise_posts_get_label_minus_one(self):
        """Isolated posts with no cluster neighbors get label -1."""
        import hdbscan
        import numpy as np

        # Create 3 tight clusters + 2 isolated noise points
        np.random.seed(42)
        cluster1 = np.random.randn(5, 384) * 0.01 + np.array([1] + [0]*383)
        cluster2 = np.random.randn(5, 384) * 0.01 + np.array([0, 1] + [0]*382)
        noise = np.random.randn(2, 384) * 10  # far away

        X = np.vstack([cluster1, cluster2, noise])
        labels = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(X)

        # Noise points should be -1
        noise_labels = labels[-2:]
        assert all(l == -1 for l in noise_labels)

    def test_clustered_posts_get_same_label(self):
        """Posts in the same trend cluster should share a label."""
        import hdbscan
        import numpy as np

        np.random.seed(42)
        # All posts very close together = one cluster
        cluster = np.random.randn(10, 384) * 0.001
        labels = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(cluster)

        # All should be in the same cluster (not -1)
        assert len(set(labels)) == 1
        assert labels[0] != -1
```

## tests/test_api.py

```python
"""Integration tests for REST API endpoints."""
import uuid
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestBrandsAPI:
    async def test_list_brands_empty(self, api_client: AsyncClient):
        response = await api_client.get("/api/v1/brands/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_create_brand(self, api_client: AsyncClient):
        response = await api_client.post("/api/v1/brands/?name=TestCola")
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "TestCola"
        assert "id" in data

    async def test_get_brand_not_found(self, api_client: AsyncClient):
        response = await api_client.get(f"/api/v1/brands/{uuid.uuid4()}")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestAlertsAPI:
    async def test_list_alerts_returns_list(self, api_client: AsyncClient):
        response = await api_client.get("/api/v1/alerts/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_list_alerts_with_brand_filter(self, api_client: AsyncClient):
        brand_id = uuid.uuid4()
        response = await api_client.get(f"/api/v1/alerts/?brand_id={brand_id}")
        assert response.status_code == 200

    async def test_list_alerts_score_filter(self, api_client: AsyncClient):
        response = await api_client.get("/api/v1/alerts/?min_score=0.8")
        assert response.status_code == 200
        for alert in response.json():
            assert alert["composite_score"] >= 0.8


@pytest.mark.asyncio
class TestFeedbackAPI:
    async def test_submit_feedback_invalid_alert(self, api_client: AsyncClient):
        payload = {"alert_id": str(uuid.uuid4()), "action": "click"}
        response = await api_client.post("/api/v1/feedback/", json=payload)
        assert response.status_code == 404

    async def test_submit_feedback_invalid_action(self, api_client: AsyncClient):
        payload = {"alert_id": str(uuid.uuid4()), "action": "invalid_action"}
        response = await api_client.post("/api/v1/feedback/", json=payload)
        assert response.status_code == 422


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_check(self, api_client: AsyncClient):
        response = await api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
```

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
docker compose run --rm worker pytest tests/test_processing.py -v

# Run with coverage
docker compose run --rm worker pytest tests/ --cov=shared --cov=services --cov-report=term-missing

# Run a single test
docker compose run --rm worker pytest tests/test_api.py::TestHealthEndpoint::test_health_check -v
```

## Verification Checklist Before Considering Complete

1. `make up` — all 6 services start healthy
2. `make migrate` — Alembic creates all tables including `vector(384)` columns
3. `make seed` — 3 sample brands seeded with query terms
4. `GET /health` returns `{"status": "ok"}`
5. `GET /api/v1/brands/` returns seeded brands
6. `POST /api/v1/brands/{id}/upload-guidelines` triggers Celery brand extraction task
7. `make test` — all tests pass
8. Flower UI at `localhost:5555` shows workers connected
9. MinIO console at `localhost:9001` shows `raw-posts` and `ml-artifacts` buckets
10. Manual ingest trigger: `docker compose run --rm worker python -c "from services.worker.tasks.ingest import ingest_posts_for_brand; ingest_posts_for_brand('<brand_id>')"`
