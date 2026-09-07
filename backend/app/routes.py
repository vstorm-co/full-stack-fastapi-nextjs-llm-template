import secrets
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo import ReturnDocument

from .config import get_settings
from .db import MongoStore
from .schemas import (
    AnalyticsEvent,
    Campaign,
    CampaignCreate,
    ChaserDispatch,
    DashboardSummary,
    Lead,
    LeadIntakeRequest,
    LeadIntakeResponse,
    OutreachAssetCreate,
    Product,
    ProductCreate,
    SpeedToLeadDemoRequest,
    SpeedToLeadDemoResponse,
    SubsystemStatus,
    Task,
    TaskCreate,
    TaskUpdate,
    TelemetryReport,
    WeeklyBudgetTier,
)
from .triage import route_lead

router = APIRouter(prefix="/api")

_operator_bearer = HTTPBearer(auto_error=False)


def store(request: Request) -> MongoStore:
    return request.app.state.store


def serialize(document: dict) -> dict:
    document = dict(document)
    document["id"] = str(document.pop("_id"))
    return document


async def require_operator(
    credentials: HTTPAuthorizationCredentials | None = Depends(_operator_bearer),
) -> None:
    """Fail-closed bearer gate for operator endpoints.

    Reads ``OPERATOR_ADMIN_KEY`` at request time; when unset the endpoint
    returns 503 so operator data can never fall open by accident.
    """
    expected = get_settings().operator_admin_key
    if not expected:
        raise HTTPException(status_code=503, detail="OPERATOR_ADMIN_KEY is not configured")
    supplied = credentials.credentials.encode("utf-8") if credentials else b""
    if not supplied or not secrets.compare_digest(supplied, expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing operator key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/health")
async def health(db: MongoStore = Depends(store)) -> dict[str, str]:
    await db.client.admin.command("ping") if db.client else None
    return {"status": "ok", "database": "connected" if db.database else "disconnected"}


@router.post("/leads/intake", response_model=LeadIntakeResponse, status_code=201)
async def create_lead_intake(
    payload: LeadIntakeRequest, db: MongoStore = Depends(store)
) -> LeadIntakeResponse:
    """Triage an inbound lead by weekly ad-spend tier (with full-stack override)
    and persist the routing state for the operator console."""
    settings = get_settings()
    tier, routing = route_lead(payload)
    document = payload.model_dump(mode="json") | {
        "_id": uuid4().hex,
        "stack": routing.stack,
        "provisioned_subsystem": routing.provisioned_subsystem,
        "trigger_chaser": routing.trigger_chaser,
        "assigned_agent": routing.assigned_agent,
        "routing_action": routing.action,
        "target_url": getattr(settings, routing.target_setting),
        "verification_state": "unverified",
        "created_at": datetime.now(UTC),
    }
    await db.collection("leads").insert_one(document)
    return LeadIntakeResponse(
        lead_id=document["_id"],
        tier=tier,
        routing_action=routing.action,
        target_url=document["target_url"],
        provisioned_subsystem=routing.provisioned_subsystem,
        assigned_agent=routing.assigned_agent,
    )


@router.post("/leads/speed-to-lead-demo", response_model=SpeedToLeadDemoResponse, status_code=201)
async def create_speed_to_lead_demo(
    payload: SpeedToLeadDemoRequest, db: MongoStore = Depends(store)
) -> SpeedToLeadDemoResponse:
    """Record a proof request without sending SMS or incurring provider spend."""
    request_id = uuid4().hex
    await db.collection("speed_demo_requests").insert_one(
        {
            "_id": request_id,
            "phone": payload.phone,
            "status": "recorded",
            "response_window_seconds": 59,
            "created_at": datetime.now(UTC),
        }
    )
    return SpeedToLeadDemoResponse(request_id=request_id)


@router.get("/leads", response_model=list[Lead])
async def list_leads(
    _operator: None = Depends(require_operator),
    tier: WeeklyBudgetTier | None = None,
    db: MongoStore = Depends(store),
) -> list[dict]:
    """Live lead feed for the operator console, optionally filtered by tier."""
    query = {"weekly_budget": tier} if tier else {}
    cursor = db.collection("leads").find(query).sort("created_at", -1).limit(200)
    return [serialize(doc) async for doc in cursor]


@router.post("/leads/{lead_id}/chaser", response_model=ChaserDispatch)
async def dispatch_chaser(
    lead_id: str,
    _operator: None = Depends(require_operator),
    db: MongoStore = Depends(store),
) -> ChaserDispatch:
    """Dispatch the AI chaser sequence (SMS + email) for a lead."""
    now = datetime.now(UTC)
    result = await db.collection("leads").find_one_and_update(
        {"_id": lead_id},
        {"$set": {"chaser_dispatched_at": now, "chaser_channels": ["sms", "email"]}},
        return_document=ReturnDocument.AFTER,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Lead not found")
    return ChaserDispatch(
        lead_id=lead_id, dispatched=True, channels=["sms", "email"], dispatched_at=now
    )


@router.get("/telemetry", response_model=TelemetryReport)
async def telemetry(
    _operator: None = Depends(require_operator),
    db: MongoStore = Depends(store),
) -> TelemetryReport:
    """Subsystem health readout: API, MongoDB, and the n8n webhook sync."""
    settings = get_settings()
    mongodb_ok = False
    if db.client is not None:
        try:
            await db.client.admin.command("ping")
            mongodb_ok = True
        except Exception:
            mongodb_ok = False
    return TelemetryReport(
        backend=SubsystemStatus(
            subsystem="fastapi_backend",
            status="online",
            detail="GrowthEngine OS API v0.1.0",
        ),
        mongodb=SubsystemStatus(
            subsystem="mongodb",
            status="online" if mongodb_ok else "offline",
            detail="Connected" if mongodb_ok else "MONGODB_URI not reachable",
        ),
        n8n_webhook=SubsystemStatus(
            subsystem="n8n_webhook",
            status="online" if settings.n8n_webhook_url else "not_configured",
            detail=(
                "Webhook registered"
                if settings.n8n_webhook_url
                else "Set N8N_WEBHOOK_URL to enable sync"
            ),
        ),
        generated_at=datetime.now(UTC),
    )


@router.get("/products", response_model=list[Product])
async def list_products(db: MongoStore = Depends(store)) -> list[dict]:
    return [serialize(doc) async for doc in db.collection("products").find().sort("created_at", -1)]


@router.post("/products", response_model=Product, status_code=201)
async def create_product(payload: ProductCreate, db: MongoStore = Depends(store)) -> dict:
    now = datetime.now(UTC)
    document = payload.model_dump(mode="json") | {"created_at": now}
    result = await db.collection("products").insert_one(document)
    document["_id"] = result.inserted_id
    return serialize(document)


@router.post("/campaigns", response_model=Campaign, status_code=201)
async def create_campaign(payload: CampaignCreate, db: MongoStore = Depends(store)) -> dict:
    if not await db.collection("products").find_one({"_id": payload.product_id}):
        raise HTTPException(status_code=404, detail="Product not found")
    document = payload.model_dump() | {"status": "active", "created_at": datetime.now(UTC)}
    document["_id"] = uuid4().hex
    await db.collection("campaigns").insert_one(document)
    return serialize(document)


@router.get("/campaigns/{campaign_id}/tasks", response_model=list[Task])
async def list_tasks(campaign_id: str, db: MongoStore = Depends(store)) -> list[dict]:
    return [serialize(doc) async for doc in db.collection("tasks").find({"campaign_id": campaign_id}).sort("created_at", -1)]


@router.post("/tasks", response_model=Task, status_code=201)
async def create_task(payload: TaskCreate, db: MongoStore = Depends(store)) -> dict:
    now = datetime.now(UTC)
    document = payload.model_dump() | {"status": "backlog", "created_at": now, "updated_at": now}
    document["_id"] = uuid4().hex
    await db.collection("tasks").insert_one(document)
    return serialize(document)


@router.patch("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, payload: TaskUpdate, db: MongoStore = Depends(store)) -> dict:
    updates = {key: value for key, value in payload.model_dump().items() if value is not None}
    updates["updated_at"] = datetime.now(UTC)
    result = await db.collection("tasks").find_one_and_update(
        {"_id": task_id}, {"$set": updates}, return_document=ReturnDocument.AFTER
    )
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return serialize(result)


@router.post("/outreach-assets", status_code=201)
async def create_outreach_asset(payload: OutreachAssetCreate, db: MongoStore = Depends(store)) -> dict:
    document = payload.model_dump() | {"created_at": datetime.now(UTC), "_id": uuid4().hex}
    await db.collection("outreach_assets").insert_one(document)
    return serialize(document)


@router.post("/analytics/events", status_code=201)
async def record_event(payload: AnalyticsEvent, db: MongoStore = Depends(store)) -> dict:
    document = payload.model_dump() | {"recorded_at": datetime.now(UTC), "_id": uuid4().hex}
    await db.collection("analytics").insert_one(document)
    return serialize(document)


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary(db: MongoStore = Depends(store)) -> DashboardSummary:
    tasks = db.collection("tasks")
    analytics = db.collection("analytics")
    return DashboardSummary(
        products=await db.collection("products").count_documents({}),
        campaigns=await db.collection("campaigns").count_documents({}),
        open_tasks=await tasks.count_documents({"status": {"$ne": "done"}}),
        leads=await analytics.count_documents({"event": "lead_captured"}),
        payment_clicks=await analytics.count_documents({"event": "payment_clicked"}),
        customers=await analytics.count_documents({"event": "customer_won"}),
    )
