"""Planner runner responsible for calling the language model and parsing output."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.core.schemas import PlannerPlan, ReflectionOutput, ToolCall

from .actor_prompt import PLANNER_PROMPT

LOGGER = logging.getLogger(__name__)

planner_prompt_template = PLANNER_PROMPT + """

MODE:
{mode}

GOAL:
{goal}

REFLECTION_CONTEXT:
{context}

CURRENT_STATE:
{state}

MEMORY_CONTEXT:
{memory}

IDENTITY:
{identity}

POLICY:
{policy}
"""


class SupportsGenerate(Protocol):
    """Protocol describing the language model client used by the planner."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        """Return the raw text completion for ``prompt``."""


class PlannerRunner:
    """Coordinates prompt building, model invocation, and response parsing."""

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        identity: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, Any]] = None,
    ):
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    def run(
        self,
        reflection: ReflectionOutput,
        current_state: Dict[str, Any],
        *,
        memory_context: Optional[List[str]] = None,
    ) -> PlannerPlan:
        context_block = json.dumps(reflection.context or {}, indent=2, ensure_ascii=False)
        state_block = json.dumps(current_state or {}, indent=2, ensure_ascii=False)
        memory_block = json.dumps(memory_context or [], indent=2, ensure_ascii=False)
        identity_block = json.dumps(self._identity, indent=2, ensure_ascii=False)
        policy_block = json.dumps(self._policy, indent=2, ensure_ascii=False)

        prompt = planner_prompt_template.format(
            mode=reflection.mode,
            goal=reflection.goal,
            context=context_block,
            state=state_block,
            memory=memory_block,
            identity=identity_block,
            policy=policy_block,
        )

        try:
            raw = self._client.generate(prompt, stop=None)
            payload = json.loads(raw)
            return self._parse_payload(payload)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Planner fallback invoked: %s", exc)
            return self._fallback_plan(reflection)

    def _parse_payload(self, payload: Dict[str, Any]) -> PlannerPlan:
        tool_calls: List[ToolCall] = []
        for tc in payload.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or tc.get("tool_name") or "").strip()
            if not name:
                continue
            arguments = tc.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(name=name, arguments=arguments))

        steps = [str(s).strip() for s in payload.get("steps", []) or [] if str(s).strip()]
        analysis = str(payload.get("analysis", "")).strip()
        response_type = str(payload.get("response_type", "text")).strip().lower()
        if response_type not in {"text", "json"}:
            response_type = "text"
        needs_risk_check = bool(payload.get("needs_risk_check", True))
        confidence = float(payload.get("confidence", 0.0) or 0.0)

        return PlannerPlan(
            analysis=analysis or "",
            steps=steps,
            tool_calls=tool_calls,
            response_type=response_type,  # type: ignore[arg-type]
            needs_risk_check=needs_risk_check,
            confidence=confidence,
        )

    def _fallback_plan(self, reflection: ReflectionOutput) -> PlannerPlan:
        base_steps = [reflection.goal] if reflection.goal else []
        analysis = reflection.goal or reflection.original_message or "General objective"
        if reflection.mode in {"chat", "knowledge"}:
            return PlannerPlan(
                analysis=analysis,
                steps=base_steps,
                tool_calls=[],
                response_type="text",
                needs_risk_check=False,
                confidence=0.0,
            )

        return PlannerPlan(
            analysis=analysis,
            steps=base_steps,
            tool_calls=[],
            response_type="text",
            needs_risk_check=True,
            confidence=0.0,
        )


__all__ = ["PlannerRunner"]
