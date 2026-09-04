"""Pure intake triage: no FastAPI or storage imports so the decision matrix is
unit-testable without a running app or database.

The route handler layers persistence and env-resolved target URLs on top of
``route_lead``; this module owns the routing rules only.
"""

from dataclasses import dataclass

from .schemas import LeadIntakeRequest, PrimaryBottleneck, WeeklyBudgetTier


@dataclass(frozen=True)
class TierRouting:
    """Static routing verdict for each weekly ad-spend tier."""

    stack: str
    provisioned_subsystem: str
    trigger_chaser: bool
    assigned_agent: str | None
    action: str
    target_setting: str


TIER_ROUTING: dict[WeeklyBudgetTier, TierRouting] = {
    WeeklyBudgetTier.TIER_SELF_SERVE: TierRouting(
        stack="tier_self_serve",
        provisioned_subsystem="self_serve_modular_tools",
        trigger_chaser=False,
        assigned_agent="self_serve_onboarding_agent",
        action="self_serve_redirect",
        target_setting="self_serve_url",
    ),
    WeeklyBudgetTier.TIER_MID_GROWTH: TierRouting(
        stack="tier_growth_pod",
        provisioned_subsystem="managed_growth_pod",
        trigger_chaser=True,
        assigned_agent="mid_tier_growth_agent",
        action="book_call",
        target_setting="growth_calendar_url",
    ),
    WeeklyBudgetTier.TIER_HIGH_TICKET: TierRouting(
        stack="tier_white_glove_agency",
        provisioned_subsystem="full_fleet_white_glove",
        trigger_chaser=True,
        assigned_agent="vip_outbound_chaser",
        action="vip_fast_track",
        target_setting="vip_calendar_url",
    ),
}


def route_lead(payload: LeadIntakeRequest) -> tuple[WeeklyBudgetTier, TierRouting]:
    """Decide the tier (with the full-operating-stack override) and its routing.

    ``weekly_budget`` normally drives the matrix, but a ``full_operating_stack``
    bottleneck always fast-tracks to the white-glove suite.
    """
    tier = payload.weekly_budget
    if payload.primary_bottleneck == PrimaryBottleneck.FULL_OPERATING_STACK:
        tier = WeeklyBudgetTier.TIER_HIGH_TICKET
    return tier, TIER_ROUTING[tier]
