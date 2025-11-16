"""Decision logic that mediates between planner, critic, and tool executor."""
from __future__ import annotations

from typing import Dict, List

from app.core.schemas import CriticReview, GovernorDecision, PlannerPlan, ToolCall


def arbitrate(
    plan: PlannerPlan,
    critic: CriticReview,
    *,
    mode: str,
    context: Dict[str, object] | None = None,
) -> GovernorDecision:
    """Return a governor verdict for the provided cycle artifacts."""

    context = context or {}
    safe_mode = mode if mode in {"chat", "knowledge", "world_query", "world_control"} else plan.mode
    metadata = {
        "mode": safe_mode,
        "critic_risk": critic.risk_level,
        "issues": critic.detected_issues,
    }
    metadata.update({key: value for key, value in context.items() if isinstance(key, str)})

    final_tool_calls = list(plan.tool_calls)

    if critic.risk_level in {"high", "unsafe"}:
        return GovernorDecision(
            verdict="reject",
            rationale=critic.notes or "Critic determined the plan was unsafe.",
            final_tool_calls=[],
            response_type=plan.response_type,
            metadata=metadata,
        )

    if safe_mode in {"chat", "knowledge"}:
        return GovernorDecision(
            verdict="approve",
            rationale="Responder-only mode; no tools required.",
            final_tool_calls=[],
            response_type=plan.response_type,
            metadata=metadata,
        )

    if safe_mode == "world_query" or context.get("query_mode"):
        read_calls = _filter_read_calls(final_tool_calls)
        if not read_calls:
            return GovernorDecision(
                verdict="request_more_data",
                rationale="Query mode requested but no read tools were planned.",
                final_tool_calls=[],
                response_type=plan.response_type,
                metadata=metadata,
            )
        return GovernorDecision(
            verdict="query_mode",
            rationale="Approved read-only system query.",
            final_tool_calls=read_calls,
            response_type=plan.response_type,
            metadata=metadata,
        )

    if safe_mode == "world_control":
        if not final_tool_calls:
            return GovernorDecision(
                verdict="request_more_data",
                rationale="Control mode requires explicit tool calls.",
                final_tool_calls=[],
                response_type=plan.response_type,
                metadata=metadata,
            )
        return GovernorDecision(
            verdict="approve",
            rationale="Control plan approved.",
            final_tool_calls=final_tool_calls,
            response_type=plan.response_type,
            metadata=metadata,
        )

    return GovernorDecision(
        verdict="approve",
        rationale="Default approval.",
        final_tool_calls=[],
        response_type=plan.response_type,
        metadata=metadata,
    )


def _filter_read_calls(calls: List[ToolCall]) -> List[ToolCall]:
    read_calls: List[ToolCall] = []
    for call in calls:
        if call.name.startswith("read"):
            read_calls.append(call)
    return read_calls


__all__ = ["arbitrate"]
