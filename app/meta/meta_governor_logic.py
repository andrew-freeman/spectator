"""Decision logic for the meta-governor applying cautious adjustments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .meta_actor_runner import MetaActorOutput, ParameterAdjustment
from .meta_critic_runner import MetaCriticOutput


@dataclass
class MetaGovernorDecision:
    decision: str
    rationale: str
    updated_params: Dict[str, Any] = field(default_factory=dict)
    applied_adjustments: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


def evaluate_meta_cycle(
    meta_actor: MetaActorOutput,
    meta_critic: MetaCriticOutput,
    current_params: Dict[str, Any],
) -> MetaGovernorDecision:
    """Evaluate whether to apply, reject, or revise meta adjustments."""

    risk = meta_critic.risk_rating.lower().strip()

    if risk in {"unsafe", "high"}:
        return MetaGovernorDecision(
            decision="reject",
            rationale="Meta-critic flagged proposal as unsafe or high risk.",
            updated_params=current_params,
            notes={"issues": meta_critic.meta_issues},
        )

    if meta_critic.confidence < 0.5 and meta_critic.meta_issues:
        return MetaGovernorDecision(
            decision="revise",
            rationale="Meta-critic expressed low confidence with outstanding issues.",
            updated_params=current_params,
            notes={"issues": meta_critic.meta_issues, "improvements": meta_critic.meta_improvements},
        )

    updated, applied = _apply_adjustments(current_params, meta_actor.parameter_adjustments)
    rationale = "Applied clamped parameter adjustments after critic approval." if applied else "No parameter adjustments proposed."

    return MetaGovernorDecision(
        decision="apply" if applied else "noop",
        rationale=rationale,
        updated_params=updated,
        applied_adjustments=applied,
        notes={
            "meta_thoughts": meta_actor.meta_thoughts,
            "critic_feedback": meta_critic.meta_improvements,
            "stability_notes": meta_critic.stability_notes,
        },
    )


def _apply_adjustments(
    current_params: Dict[str, Any],
    adjustments: Dict[str, ParameterAdjustment],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    params_copy = deepcopy(current_params)
    applied: Dict[str, float] = {}

    for name, change in adjustments.items():
        clamped_delta = _clamp_delta(change.delta)
        base_value = float(params_copy.get(name, 0.0))
        params_copy[name] = round(base_value + clamped_delta, 6)
        if clamped_delta:
            applied[name] = clamped_delta

    return params_copy, applied


def _clamp_delta(delta: float) -> float:
    maximum_step = 0.05
    if delta > maximum_step:
        return maximum_step
    if delta < -maximum_step:
        return -maximum_step
    return delta


__all__ = ["MetaGovernorDecision", "evaluate_meta_cycle"]
