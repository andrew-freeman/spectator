"""Critic runner that validates planner outputs for safety and consistency."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import CriticOutput, PlannerPlan

from .critic_prompt import CRITIC_PROMPT

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class CriticRunner:
    """Prepare prompt, call model, and parse structured critic feedback."""

    def __init__(self, client: SupportsGenerate, *, identity: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    def run_plan(self, plan: PlannerPlan, *, mode: str) -> CriticOutput:
        prompt = CRITIC_PROMPT.format(
            plan=json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
            mode=mode,
            identity=json.dumps(self._identity, indent=2, ensure_ascii=False),
            policy=json.dumps(self._policy, indent=2, ensure_ascii=False),
        )
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = self._parse_json(raw)
            return self._build_output(payload)
        except Exception as exc:
            LOGGER.warning("Critic fallback invoked: %s", exc)
            return CriticOutput(risk_level="low", confidence=0.0, detected_issues=[], notes="Critic fallback invoked.")

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        snippet = raw.strip()
        first = snippet.find("{")
        last = snippet.rfind("}")
        if first != -1 and last != -1:
            snippet = snippet[first : last + 1]
        return json.loads(snippet)

    def _build_output(self, payload: Dict[str, Any]) -> CriticOutput:
        risk = str(payload.get("risk_level", "low")).strip().lower()
        if risk not in {"low", "medium", "high", "unsafe"}:
            risk = "low"
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        detected = [str(item).strip() for item in payload.get("detected_issues", []) or [] if str(item).strip()]
        notes = str(payload.get("notes", "")).strip()
        return CriticOutput(
            risk_level=risk,  # type: ignore[arg-type]
            confidence=confidence,
            detected_issues=detected,
            notes=notes,
        )


__all__ = ["CriticRunner"]
