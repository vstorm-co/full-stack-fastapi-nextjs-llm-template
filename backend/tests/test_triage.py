"""Triage matrix unit tests for the GrowthEngine intake gatekeeper.

Covers the three weekly-spend tiers and the ``full_operating_stack`` override.
The decision logic lives in ``app.triage`` (no FastAPI/database imports) and
target URLs are resolved through the real ``Settings`` defaults so a change to
either the matrix or the routing config fails loudly here.
"""

from app.config import get_settings
from app.schemas import LeadIntakeRequest, PrimaryBottleneck, WeeklyBudgetTier
from app.triage import route_lead


def _lead(
    budget: WeeklyBudgetTier,
    bottleneck: PrimaryBottleneck = PrimaryBottleneck.LEAD_GENERATION,
) -> LeadIntakeRequest:
    return LeadIntakeRequest(
        name="Test Operator",
        email="operator@example.com",
        phone="+1 555 010 0000",
        company_name="Test Co",
        primary_bottleneck=bottleneck,
        weekly_budget=budget,
    )


def _target_url(budget: WeeklyBudgetTier) -> str:
    """Resolve the routing target from env settings (defaults in tests)."""
    _, routing = route_lead(_lead(budget))
    return getattr(get_settings(), routing.target_setting)


def test_under_250_routes_to_self_serve_redirect() -> None:
    tier, routing = route_lead(_lead(WeeklyBudgetTier.TIER_SELF_SERVE))

    assert tier == WeeklyBudgetTier.TIER_SELF_SERVE
    assert routing.stack == "tier_self_serve"
    assert routing.provisioned_subsystem == "self_serve_modular_tools"
    assert routing.action == "self_serve_redirect"
    assert routing.trigger_chaser is False
    assert routing.assigned_agent == "self_serve_onboarding_agent"
    assert _target_url(WeeklyBudgetTier.TIER_SELF_SERVE) == "/portal/onboarding"


def test_mid_tier_routes_to_growth_booking_call() -> None:
    tier, routing = route_lead(_lead(WeeklyBudgetTier.TIER_MID_GROWTH))

    assert tier == WeeklyBudgetTier.TIER_MID_GROWTH
    assert routing.stack == "tier_growth_pod"
    assert routing.provisioned_subsystem == "managed_growth_pod"
    assert routing.action == "book_call"
    assert routing.trigger_chaser is True
    assert routing.assigned_agent == "mid_tier_growth_agent"
    assert _target_url(WeeklyBudgetTier.TIER_MID_GROWTH) == "/booking/growth"


def test_high_tier_routes_to_vip_fast_track() -> None:
    tier, routing = route_lead(_lead(WeeklyBudgetTier.TIER_HIGH_TICKET))

    assert tier == WeeklyBudgetTier.TIER_HIGH_TICKET
    assert routing.stack == "tier_white_glove_agency"
    assert routing.provisioned_subsystem == "full_fleet_white_glove"
    assert routing.action == "vip_fast_track"
    assert routing.trigger_chaser is True
    assert routing.assigned_agent == "vip_outbound_chaser"
    assert _target_url(WeeklyBudgetTier.TIER_HIGH_TICKET) == "/booking/vip-fast-track"


def test_full_operating_stack_overrides_low_budget_to_vip() -> None:
    """A full-stack bottleneck fast-tracks even when weekly spend is low."""
    lead = _lead(WeeklyBudgetTier.TIER_SELF_SERVE, PrimaryBottleneck.FULL_OPERATING_STACK)

    tier, routing = route_lead(lead)

    assert tier == WeeklyBudgetTier.TIER_HIGH_TICKET
    assert routing.action == "vip_fast_track"
    assert routing.stack == "tier_white_glove_agency"
    assert routing.trigger_chaser is True


def test_all_three_tiers_resolve_distinct_destinations() -> None:
    urls = {
        WeeklyBudgetTier.TIER_SELF_SERVE: "/portal/onboarding",
        WeeklyBudgetTier.TIER_MID_GROWTH: "/booking/growth",
        WeeklyBudgetTier.TIER_HIGH_TICKET: "/booking/vip-fast-track",
    }
    for budget, expected in urls.items():
        assert _target_url(budget) == expected
