"""Runner coordinating meta-actor reasoning cycles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .meta_actor_prompt import build_meta_actor_prompt


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class ParameterAdjustment:
    delta: float
    justification: str


@dataclass
class MetaActorOutput:
    meta_thoughts: List[str]
    cognitive_strategy: Dict[str, Any]
    parameter_adjustments: Dict[str, ParameterAdjustment]
    meta_improvements: List[str]
    assumptions: List[str]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "MetaActorOutput":
        adjustments = {
            key: ParameterAdjustment(delta=float(value.get("delta", 0.0)), justification=value.get("justification", ""))
            for key, value in (payload.get("parameter_adjustments") or {}).items()
        }
        return cls(
            meta_thoughts=list(payload.get("meta_thoughts", [])),
            cognitive_strategy=dict(payload.get("cognitive_strategy", {})),
            parameter_adjustments=adjustments,
            meta_improvements=list(payload.get("meta_improvements", [])),
            assumptions=list(payload.get("assumptions", [])),
        )


class MetaActorRunner:
    def __init__(self, client: SupportsGenerate):
        self._client = client

    def run(
        self,
        current_params: Dict[str, Any],
        recent_decisions: List[Dict[str, Any]],
        meta_cycle: int,
        system_limits: Optional[Dict[str, Any]] = None,
    ) -> MetaActorOutput:
        prompt = build_meta_actor_prompt(
            current_params=current_params,
            recent_decisions=recent_decisions,
            meta_cycle=meta_cycle,
            system_limits=system_limits,
        )
        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)
        return MetaActorOutput.from_json(payload)


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Meta-actor returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["MetaActorRunner", "MetaActorOutput", "ParameterAdjustment"]
