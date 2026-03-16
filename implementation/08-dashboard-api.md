# Dashboard API — FastAPI REST + WebSocket

## services/api/main.py

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import alerts, feedback, brands, ws
from shared.db import async_engine
from shared.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (migrations handle schema, this is a safety net)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await async_engine.dispose()


app = FastAPI(title="CPG Trend Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(brands.router, prefix="/api/v1/brands", tags=["brands"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
app.include_router(ws.router, prefix="/ws", tags=["websocket"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

## services/api/routers/brands.py

```python
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from shared.db import get_db
from shared.models import Brand
from shared.schemas import BrandOut
from shared.utils.s3 import get_s3_client
from shared.settings import Settings

router = APIRouter()
settings = Settings()


@router.get("/", response_model=list[BrandOut])
async def list_brands(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Brand).order_by(Brand.name))
    return result.scalars().all()


@router.get("/{brand_id}", response_model=BrandOut)
async def get_brand(brand_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    brand = await db.get(Brand, brand_id)
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    return brand


@router.post("/", response_model=BrandOut, status_code=201)
async def create_brand(name: str, db: AsyncSession = Depends(get_db)):
    brand = Brand(name=name)
    db.add(brand)
    await db.commit()
    await db.refresh(brand)
    return brand


@router.post("/{brand_id}/upload-guidelines")
async def upload_brand_guidelines(
    brand_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload brand guidelines PDF. Triggers async brand profile extraction."""
    brand = await db.get(Brand, brand_id)
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Store PDF to S3
    s3_key = f"guidelines/{brand_id}/{file.filename}"
    s3 = get_s3_client()
    content = await file.read()
    s3.put_object(
        Bucket=settings.s3_artifacts_bucket,
        Key=s3_key,
        Body=content,
        ContentType="application/pdf",
    )

    brand.pdf_s3_key = s3_key
    await db.commit()

    # Trigger async extraction
    from services.worker.tasks.ingest import extract_brand_profile
    extract_brand_profile.delay(str(brand_id), s3_key)

    return {"status": "processing", "s3_key": s3_key}
```

## services/api/routers/alerts.py

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
import uuid

from shared.db import get_db
from shared.models import Alert, Post
from shared.schemas import AlertOut

router = APIRouter()


@router.get("/", response_model=list[AlertOut])
async def list_alerts(
    brand_id: uuid.UUID | None = None,
    since_hours: int = Query(24, ge=1, le=168),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List recent alerts with optional brand filter and score threshold."""
    since = datetime.utcnow() - timedelta(hours=since_hours)

    conditions = [Alert.created_at >= since, Alert.composite_score >= min_score]
    if brand_id:
        conditions.append(Alert.brand_id == brand_id)

    result = await db.execute(
        select(Alert, Post.text, Post.platform)
        .join(Post, Post.id == Alert.post_id)
        .where(and_(*conditions))
        .order_by(Alert.composite_score.desc())
        .limit(limit)
    )
    rows = result.all()

    return [
        AlertOut(
            id=alert.id,
            brand_id=alert.brand_id,
            post_id=alert.post_id,
            post_text=post_text,
            platform=platform,
            composite_score=alert.composite_score,
            relevance_score=alert.relevance_score,
            velocity_score=alert.velocity_score,
            why_relevant=alert.why_relevant,
            cluster_id=alert.cluster_id,
            created_at=alert.created_at,
        )
        for alert, post_text, platform in rows
    ]


@router.get("/{alert_id}", response_model=AlertOut)
async def get_alert(alert_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException
    result = await db.execute(
        select(Alert, Post.text, Post.platform)
        .join(Post, Post.id == Alert.post_id)
        .where(Alert.id == alert_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert, post_text, platform = row
    return AlertOut(
        id=alert.id,
        brand_id=alert.brand_id,
        post_id=alert.post_id,
        post_text=post_text,
        platform=platform,
        composite_score=alert.composite_score,
        relevance_score=alert.relevance_score,
        velocity_score=alert.velocity_score,
        why_relevant=alert.why_relevant,
        cluster_id=alert.cluster_id,
        created_at=alert.created_at,
    )
```

## services/api/routers/feedback.py

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from shared.db import get_db
from shared.models import Alert, FeedbackEvent
from shared.schemas import FeedbackIn

router = APIRouter()


@router.post("/", status_code=201)
async def submit_feedback(payload: FeedbackIn, db: AsyncSession = Depends(get_db)):
    """Record brand manager feedback on an alert."""
    alert = await db.get(Alert, payload.alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    event = FeedbackEvent(
        alert_id=payload.alert_id,
        brand_id=alert.brand_id,
        action=payload.action,
        dwell_ms=payload.dwell_ms,
    )
    db.add(event)
    await db.commit()
    return {"status": "recorded", "action": payload.action}
```

## services/api/routers/ws.py

```python
"""
WebSocket endpoint for real-time alert push.
Brand managers connect to /ws/alerts/{brand_id} and receive
new alerts as they are assembled.

Architecture: Celery publishes alert events to a Redis pub/sub channel.
FastAPI WebSocket handler subscribes and forwards to connected clients.
"""
import asyncio
import json
import uuid
import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from shared.settings import Settings

router = APIRouter()
settings = Settings()


@router.websocket("/alerts/{brand_id}")
async def alerts_websocket(websocket: WebSocket, brand_id: uuid.UUID):
    await websocket.accept()

    r = aioredis.from_url(settings.redis_url)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"alerts:{brand_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await websocket.send_text(data)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"alerts:{brand_id}")
        await r.aclose()
```

## Publishing alerts to WebSocket channel

In `services/worker/tasks/assemble.py`, after inserting alerts, publish to Redis:

```python
# After creating alerts in assemble_alerts task, add:
import redis as sync_redis
import json

r = sync_redis.from_url(settings.redis_url)
for post_id, scores in to_create:
    alert_data = {
        "post_id": post_id,
        "composite_score": scores["composite"],
        "why_relevant": scores["why_relevant"],
        "brand_id": brand_id,
    }
    r.publish(f"alerts:{brand_id}", json.dumps(alert_data))
```

## services/api/requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
sqlalchemy==2.0.35
asyncpg==0.29.0
pydantic==2.7.0
pydantic-settings==2.3.0
redis==5.0.8
boto3==1.35.0
python-multipart==0.0.9
websockets==13.0
celery[redis]==5.4.0
```
