from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, HttpUrl

# Mutable defaults are avoided by constructing these fields per request.


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url: HttpUrl | None = None
    value_proposition: str = Field(min_length=1, max_length=2000)
    persona: str = Field(min_length=1, max_length=1000)
    pricing: str = Field(min_length=1, max_length=300)


class Product(ProductCreate):
    id: str
    created_at: datetime


class CampaignCreate(BaseModel):
    product_id: str
    name: str = Field(default="Launch campaign", min_length=1, max_length=120)
    goal: str = Field(default="Find qualified paying customers", max_length=500)


class Campaign(CampaignCreate):
    id: str
    status: str
    created_at: datetime


class TaskCreate(BaseModel):
    campaign_id: str
    title: str
    agent: Literal["strategist", "outreach", "content", "distribution", "analytics"]
    description: str = ""
    priority: Literal["low", "medium", "high"] = "medium"


class TaskUpdate(BaseModel):
    status: Literal["backlog", "in_progress", "review", "done"] | None = None
    priority: Literal["low", "medium", "high"] | None = None


class Task(TaskCreate):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime


class OutreachAssetCreate(BaseModel):
    campaign_id: str
    channel: Literal["email", "linkedin", "community", "ad"]
    title: str
    body: str
    approval_required: bool = True


class AnalyticsEvent(BaseModel):
    campaign_id: str
    event: Literal["lead_captured", "demo_requested", "payment_clicked", "customer_won"]
    value: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardSummary(BaseModel):
    products: int
    campaigns: int
    open_tasks: int
    leads: int
    payment_clicks: int
    customers: int


class WeeklyBudgetTier(StrEnum):
    """Weekly ad-spend brackets that drive intake routing decisions."""

    TIER_SELF_SERVE = "under_250"  # Under $250/week -> self-serve suite off-ramp
    TIER_MID_GROWTH = "250_to_1250"  # $250-$1,250/week -> core growth pod
    TIER_HIGH_TICKET = "over_1250"  # $1,250+/week -> VIP white-glove pipeline


class PrimaryBottleneck(StrEnum):
    """Question 1: the operational hurdle the lead needs solved."""

    LEAD_GENERATION = "lead_generation"
    OPERATIONAL_OVERHEAD = "operational_overhead"
    LOCAL_FOOT_TRAFFIC = "local_foot_traffic"
    RETENTION_AND_CHURN = "retention_and_churn"
    FULL_OPERATING_STACK = "full_operating_stack"


class RequestedModule(StrEnum):
    """Fleet subsystems a lead can self-provision during intake."""

    VIDEO_ADS = "video_ads"  # Influencer Echo / video engine
    PRINT_COLLATERAL = "print_collateral"  # Print-For-You studio
    EMAIL_RETENTION = "email_retention"  # OmniTrickle mail engine
    LOCAL_POS_ATTRIBUTION = "local_pos_attribution"  # OmniLocal OS


class LeadIntakeRequest(BaseModel):
    """Dynamic gatekeeper intake: bottleneck, budget, and requested fleet modules."""

    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    phone: str = Field(min_length=7, max_length=32)
    company_name: str = Field(min_length=1, max_length=160)
    primary_bottleneck: PrimaryBottleneck
    weekly_budget: WeeklyBudgetTier
    requested_modules: list[RequestedModule] = Field(default_factory=list)
    # Kept optional so legacy intake payloads remain valid.
    timeline_to_scale: str = Field(default="", max_length=500)
    # White-glove fast-track booking details (optional for standard intake).
    preferred_call_time: str = Field(default="", max_length=200)
    target_start_date: str = Field(default="", max_length=100)


class LeadIntakeResponse(BaseModel):
    """Triage verdict returned so the frontend can follow target_url."""

    lead_id: str
    tier: WeeklyBudgetTier
    routing_action: Literal["self_serve_redirect", "book_call", "vip_fast_track"]
    target_url: str | None = None
    provisioned_subsystem: str | None = None
    assigned_agent: str | None = None


class SpeedToLeadDemoRequest(BaseModel):
    """Consent is intentionally not implied: this records a demo request only."""

    phone: str = Field(min_length=7, max_length=32)


class SpeedToLeadDemoResponse(BaseModel):
    """Persisted proof request; no external message is sent by this endpoint."""

    request_id: str
    status: Literal["recorded"]
    response_window_seconds: int = 59


class Lead(LeadIntakeRequest):
    """Persisted lead record surfaced to the operator console."""

    id: str
    stack: str
    provisioned_subsystem: str | None = None
    trigger_chaser: bool
    routing_action: str
    target_url: str | None = None
    assigned_agent: str | None = None
    verification_state: str
    chaser_dispatched_at: datetime | None = None
    created_at: datetime


class ChaserDispatch(BaseModel):
    """Confirmation that an AI chaser sequence was dispatched for a lead."""

    lead_id: str
    dispatched: bool
    channels: list[str]
    dispatched_at: datetime


class SubsystemStatus(BaseModel):
    """Health readout for one fleet subsystem."""

    subsystem: str
    status: Literal["online", "degraded", "offline", "not_configured"]
    detail: str


class TelemetryReport(BaseModel):
    """Operator-facing telemetry for the service fleet."""

    backend: SubsystemStatus
    mongodb: SubsystemStatus
    n8n_webhook: SubsystemStatus
    generated_at: datetime
