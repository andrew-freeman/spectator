"""Planner runner responsible for structured planning output."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.core.schemas import PlannerPlan, ReflectionOutput, ToolCall

from .actor_prompt import PLANNER_PROMPT

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class PlannerRunner:
    """Call the LLM planner and normalise its JSON response."""

    def __init__(self, client: SupportsGenerate, *, identity: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
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
        memory_block = memory_context or []
        prompt = PLANNER_PROMPT.format(
            reflection=json.dumps(reflection.to_dict(), indent=2, ensure_ascii=False),
            state=json.dumps(current_state or {}, indent=2, ensure_ascii=False),
            memory=json.dumps(memory_block, indent=2, ensure_ascii=False),
            identity=json.dumps(self._identity, indent=2, ensure_ascii=False),
            policy=json.dumps(self._policy, indent=2, ensure_ascii=False),
        )
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = self._parse_json(raw)
            return self._build_plan(payload, fallback_mode=reflection.mode)
        except Exception as exc:
            LOGGER.warning("Planner fallback invoked: %s", exc)
            return self._fallback_plan(reflection)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        snippet = raw.strip()
        first = snippet.find("{")
        last = snippet.rfind("}")
        if first != -1 and last != -1:
            snippet = snippet[first : last + 1]
        return json.loads(snippet)

    def _build_plan(self, payload: Dict[str, Any], *, fallback_mode: str) -> PlannerPlan:
        mode = str(payload.get("mode", fallback_mode or "chat")).strip().lower()
        if mode not in {"chat", "knowledge", "world_query", "world_control"}:
            mode = fallback_mode if fallback_mode in {"chat", "knowledge", "world_query", "world_control"} else "chat"
        analysis = str(payload.get("analysis", "")).strip()
        steps = [str(step).strip() for step in payload.get("steps", []) or [] if str(step).strip()]
        tool_calls = self._parse_tool_calls(payload.get("tool_calls", []))
        response_type = str(payload.get("response_type", "text")).strip().lower()
        if response_type not in {"text", "json"}:
            response_type = "text"
        needs_risk_check = bool(payload.get("needs_risk_check", mode in {"world_query", "world_control"}))
        confidence = float(payload.get("confidence", 0.0) or 0.0)

        # Enforce mode-specific expectations
        if mode == "knowledge":
            tool_calls = []
            needs_risk_check = False
        if mode == "chat":
            tool_calls = []
        return PlannerPlan(
            mode=mode,  # type: ignore[arg-type]
            analysis=analysis,
            steps=steps,
            tool_calls=tool_calls,
            response_type=response_type,  # type: ignore[arg-type]
            needs_risk_check=needs_risk_check,
            confidence=confidence,
        )

    def _parse_tool_calls(self, raw_calls: Any) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        if not isinstance(raw_calls, list):
            return tool_calls
        for entry in raw_calls:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("tool_name") or "").strip()
            if not name:
                continue
            arguments = entry.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(name=name, arguments=arguments))
        return tool_calls

    def _fallback_plan(self, reflection: ReflectionOutput) -> PlannerPlan:
        analysis = reflection.goal or "Engage user"
        steps = [reflection.goal] if reflection.goal else []
        tool_calls: List[ToolCall] = []
        if reflection.mode in {"world_query", "world_control"}:
            analysis = reflection.goal or "Perform requested system action"
        return PlannerPlan(
            mode=reflection.mode,  # type: ignore[arg-type]
            analysis=analysis,
            steps=steps,
            tool_calls=tool_calls,
            response_type="text",
            needs_risk_check=reflection.mode in {"world_query", "world_control"},
            confidence=0.0,
        )


__all__ = ["PlannerRunner"]
