"""Utility helpers for governor arbitration and tool execution prep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.actor.actor_runner import ActorOutput, ToolCall
from app.critic.critic_runner import CriticOutput


@dataclass
class MergedPlan:
    """Container describing the reconciled plan between actor and critic."""

    steps: List[str]
    notes: Optional[str] = None
    tool_calls: List[ToolCall] = None


def normalise_plan(steps: Iterable[str]) -> List[str]:
    """Return a list of trimmed plan steps, removing falsy entries."""

    return [step.strip() for step in steps if isinstance(step, str) and step.strip()]


def summarise_disagreements(actor: ActorOutput, critic: CriticOutput) -> Dict[str, Any]:
    """Produce a simple dictionary of disagreement metadata."""

    return {
        "issues": critic.detected_issues,
        "recommendations": critic.recommendations,
        "actor_information_gaps": actor.information_gaps,
    }


def merge_plans(actor: ActorOutput, critic: CriticOutput) -> MergedPlan:
    """Combine actor plan with critic recommendations deterministically."""

    steps = normalise_plan(actor.plan)
    for rec in critic.recommendations:
        if rec not in steps:
            steps.append(rec)

    notes = None
    if critic.detected_issues:
        notes = "Critic flagged issues: " + "; ".join(critic.detected_issues)

    return MergedPlan(steps=steps, notes=notes, tool_calls=actor.tool_calls)


__all__ = ["MergedPlan", "merge_plans", "normalise_plan", "summarise_disagreements"]
