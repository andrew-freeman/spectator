"""Decision logic that mediates between planner, critic, and tool executor."""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.schemas import CriticOutput, GovernorDecision, PlannerPlan


def arbitrate(
    planner_plan: PlannerPlan,
    critic_output: CriticOutput,
    *,
    mode: str,
    context: Optional[Dict[str, Any]] = None,
) -> GovernorDecision:
    """Return a governor verdict for the provided cycle artifacts."""

    context = context or {}
    metadata = {"mode": mode, "critic_risk": critic_output.risk}
    metadata.update({k: v for k, v in context.items() if isinstance(k, str)})

    final_tool_calls = critic_output.adjusted_tool_calls or planner_plan.tool_calls

    if critic_output.risk in {"high", "unsafe"}:
        rationale = critic_output.notes or "Critic marked the plan as unsafe."
        if critic_output.issues:
            rationale += " Issues: " + "; ".join(critic_output.issues)
        return GovernorDecision(
            verdict="reject",
            rationale=rationale,
            final_tool_calls=[],
            final_response_type="text",
            metadata=metadata,
        )

    if mode in {"chat", "knowledge"}:
        return GovernorDecision(
            verdict="approve",
            rationale="Responder-only mode; no tools required.",
            final_tool_calls=[],
            final_response_type="text",
            metadata=metadata,
        )

    if not final_tool_calls:
        rationale = "Plan lacked required tool calls for mode=" + mode
        verdict = "request_more_data" if mode in {"world_query", "world_control"} else "approve"
        return GovernorDecision(
            verdict=verdict,
            rationale=rationale,
            final_tool_calls=[],
            final_response_type=planner_plan.response_type,
            metadata=metadata,
        )

    return GovernorDecision(
        verdict="approve",
        rationale="Plan approved for execution.",
        final_tool_calls=final_tool_calls,
        final_response_type=planner_plan.response_type,
        metadata=metadata,
    )


__all__ = ["arbitrate", "GovernorDecision"]
