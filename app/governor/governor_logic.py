"""Deterministic arbitration logic for the base-layer governor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.actor.actor_runner import ActorOutput, ToolCall
from app.critic.critic_runner import CriticOutput

from .helper_functions import MergedPlan, merge_plans, summarise_disagreements


@dataclass
class GovernorDecision:
    """Structured result of the governor arbitration."""

    verdict: str
    rationale: str
    plan: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def arbitrate(
    actor: ActorOutput,
    critic: CriticOutput,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> GovernorDecision:
    """Deterministically decide how to proceed based on actor/critic outputs."""

    context = context or {}

    if context.get("chat_mode"):
        return GovernorDecision(
            verdict="chat_mode",
            rationale="Chat mode conversation; skipping tool execution.",
            plan=[],
            tool_calls=[],
            metadata={"notes": "Chat intent"},
        )

    if context.get("query_mode"):
        return GovernorDecision(
            verdict="query_mode",
            rationale="Information request; executing only actor tool calls.",
            plan=actor.plan,
            tool_calls=actor.tool_calls,
            metadata={},
        )

    def _finalise(decision: GovernorDecision) -> GovernorDecision:
        if context.get("force_action") and decision.verdict in {"request_more_data", "defer_to_critic"}:
            decision.verdict = "approve"
            decision.plan = actor.plan
            decision.tool_calls = actor.tool_calls
        return decision

    # Missing data guard.
    if not actor.plan or not actor.analysis:
        return _finalise(
            GovernorDecision(
                verdict="request_more_data",
                rationale="Actor response was incomplete; requesting more context.",
                metadata=summarise_disagreements(actor, critic),
            )
        )

    risk = critic.risk_level.lower().strip()

    if risk == "unsafe" or risk == "high":
        return _finalise(
            GovernorDecision(
                verdict="defer_to_critic",
                rationale="Critic identified unsafe or high-risk behaviour.",
                metadata=summarise_disagreements(actor, critic),
            )
        )

    if critic.confidence < 0.4:
        return _finalise(
            GovernorDecision(
                verdict="trust_actor",
                rationale="Critic confidence too low; defaulting to actor plan.",
                plan=actor.plan,
                tool_calls=actor.tool_calls,
                metadata=summarise_disagreements(actor, critic),
            )
        )

    if critic.detected_issues:
        merged: MergedPlan = merge_plans(actor, critic)
        return _finalise(
            GovernorDecision(
                verdict="merge",
                rationale="Resolved partial mismatch by merging actor plan with critic feedback.",
                plan=merged.steps,
                tool_calls=merged.tool_calls or [],
                metadata={**summarise_disagreements(actor, critic), "notes": merged.notes},
            )
        )

    return _finalise(
        GovernorDecision(
            verdict="trust_actor",
            rationale="Critic found no issues and maintained adequate confidence.",
            plan=actor.plan,
            tool_calls=actor.tool_calls,
            metadata=summarise_disagreements(actor, critic),
        )
    )


__all__ = ["GovernorDecision", "arbitrate"]
