"""Utility helpers for governor arbitration and tool execution prep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.core.schemas import CriticReview, PlannerPlan, ToolCall


@dataclass
class MergedPlan:
    """Container describing the reconciled plan between planner and critic."""

    steps: List[str]
    notes: Optional[str] = None
    tool_calls: List[ToolCall] | None = None


def normalise_plan(steps: Iterable[str]) -> List[str]:
    """Return a list of trimmed plan steps, removing falsy entries."""

    return [step.strip() for step in steps if isinstance(step, str) and step.strip()]


def summarise_disagreements(plan: PlannerPlan, critic: CriticReview) -> Dict[str, Any]:
    """Produce a simple dictionary of disagreement metadata."""

    return {
        "issues": critic.detected_issues,
        "planner_steps": plan.steps,
        "critic_notes": critic.notes,
    }


def merge_plans(plan: PlannerPlan, critic: CriticReview) -> MergedPlan:
    """Combine planner steps with critic issues deterministically."""

    steps = normalise_plan(plan.steps)
    notes = None
    if critic.detected_issues:
        notes = "Critic flagged issues: " + "; ".join(critic.detected_issues)

    return MergedPlan(steps=steps, notes=notes, tool_calls=plan.tool_calls)


__all__ = ["MergedPlan", "merge_plans", "normalise_plan", "summarise_disagreements"]
