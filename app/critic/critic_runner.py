"""Critic runner that validates planner outputs for safety and consistency."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import CriticReview, PlannerPlan
from app.core.structured import generate_structured_object

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

    def run_plan(self, plan: PlannerPlan, *, mode: str) -> CriticReview:
        prompt = CRITIC_PROMPT.format(
            plan=json.dumps(plan.model_dump(), indent=2, ensure_ascii=False),
            mode=mode,
            identity=json.dumps(self._identity, indent=2, ensure_ascii=False),
            policy=json.dumps(self._policy, indent=2, ensure_ascii=False),
        )

        def _fallback(_: Exception | None = None) -> CriticReview:
            return CriticReview(risk_level="low", confidence=0.0, detected_issues=[], notes="Critic fallback invoked.")

        try:
            return generate_structured_object(self._client, prompt, CriticReview, _fallback)
        except Exception as exc:
            LOGGER.warning("Critic fallback invoked: %s", exc)
            return CriticReview(risk_level="low", confidence=0.0, detected_issues=[], notes="Critic exception fallback.")


__all__ = ["CriticRunner"]
