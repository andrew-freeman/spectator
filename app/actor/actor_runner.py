"""Planner runner responsible for calling the language model and parsing output."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import Plan, PreprocessorOutput, ToolCall

from .actor_prompt import PLANNER_PROMPT

planner_prompt_template = PLANNER_PROMPT + """

MODE:
{mode}

GOAL:
{goal}

CONTEXT:
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

    def run(self, preprocessed: PreprocessorOutput, current_state: Dict[str, Any]) -> Plan:
        context_block = json.dumps(preprocessed.context or {}, indent=2)
        state_block = json.dumps(current_state or {}, indent=2)
        memory_block = json.dumps(preprocessed.memory_context or [], indent=2)
        identity_block = json.dumps(self._identity, indent=2)
        policy_block = json.dumps(self._policy, indent=2)

        prompt = planner_prompt_template.format(
            mode=preprocessed.mode,
            goal=preprocessed.goal,
            context=context_block,
            state=state_block,
            memory=memory_block,
            identity=identity_block,
            policy=policy_block,
        )

        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)

        tool_calls = []
        for tc in payload.get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get("tool_name") or tc.get("name") or ""
            tool_calls.append(
                ToolCall(tool_name=tool_name, arguments=tc.get("arguments", {}) or {})
            )

        steps = [str(s).strip() for s in payload.get("steps", []) if str(s).strip()]
        analysis = str(payload.get("analysis", "")).strip()
        response_type = str(payload.get("response_type", "text")).strip().lower()
        if response_type not in {"text", "json"}:
            response_type = "text"
        needs_risk_check = bool(payload.get("needs_risk_check", True))
        confidence = float(payload.get("confidence", 0.0) or 0.0)

        return Plan(
            analysis=analysis,
            steps=steps,
            tool_calls=tool_calls,
            response_type=response_type,  # type: ignore[arg-type]
            needs_risk_check=needs_risk_check,
            confidence=confidence,
        )


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse the model response and surface helpful errors."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Planner returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["PlannerRunner"]
