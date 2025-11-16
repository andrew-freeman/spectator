"""Critic runner that validates actor outputs for safety and consistency."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .critic_prompt import build_critic_prompt


class SupportsGenerate(Protocol):
    """Protocol representing the shared language model interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class CriticOutput:
    """Structured payload produced by the critic."""

    evaluation: str
    detected_issues: List[str]
    risk_level: str
    confidence: float
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "CriticOutput":
        return cls(
            evaluation=payload.get("evaluation", ""),
            detected_issues=list(payload.get("detected_issues", [])),
            risk_level=str(payload.get("risk_level", "low")),
            confidence=float(payload.get("confidence", 0.0)),
            recommendations=list(payload.get("recommendations", [])),
        )


class CriticRunner:
    """Prepare prompt, call model, and parse structured critic feedback."""

    def __init__(self, client: SupportsGenerate, *, identity: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    def run(
        self,
        actor_payload: Dict[str, Any],
        safety_policies: Optional[List[str]] = None,
    ) -> CriticOutput:
        prompt = build_critic_prompt(
            actor_payload,
            safety_policies=safety_policies,
            identity=self._identity,
            policy=self._policy,
        )
        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)
        return CriticOutput.from_json(payload)


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Critic returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["CriticRunner", "CriticOutput"]
