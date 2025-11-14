"""Runner for evaluating meta-actor proposals via the meta-critic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .meta_critic_prompt import build_meta_critic_prompt


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class MetaCriticOutput:
    meta_evaluation: str
    meta_issues: List[str]
    risk_rating: str
    confidence: float
    meta_improvements: List[str]
    stability_notes: Optional[str]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "MetaCriticOutput":
        return cls(
            meta_evaluation=payload.get("meta_evaluation", ""),
            meta_issues=list(payload.get("meta_issues", [])),
            risk_rating=str(payload.get("risk_rating", "low")),
            confidence=float(payload.get("confidence", 0.0)),
            meta_improvements=list(payload.get("meta_improvements", [])),
            stability_notes=payload.get("stability_notes"),
        )


class MetaCriticRunner:
    def __init__(self, client: SupportsGenerate):
        self._client = client

    def run(self, meta_actor_payload: Dict[str, Any], current_params: Dict[str, Any]) -> MetaCriticOutput:
        prompt = build_meta_critic_prompt(meta_actor_payload, current_params)
        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)
        return MetaCriticOutput.from_json(payload)


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Meta-critic returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["MetaCriticRunner", "MetaCriticOutput"]
