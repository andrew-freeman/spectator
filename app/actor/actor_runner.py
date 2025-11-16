"""Planner runner responsible for structured planning output."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.core.schemas import PlannerPlan, ReflectionOutput, ToolCall

from .actor_prompt import PLANNER_PROMPT

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Protocol for the LLM client used by the planner."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class PlannerRunner:
    """Call the LLM planner and normalise its JSON response into a PlannerPlan."""

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        identity: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        """Build the planner prompt, call the LLM, and parse the result."""
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
            return self._build_plan(payload, reflection=reflection)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Planner fallback invoked due to error: %s", exc)
            return self._fallback_plan(reflection)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Extract the first top-level JSON object from a raw completion."""
        snippet = raw.strip()
        first = snippet.find("{")
        last = snippet.rfind("}")
        if first != -1 and last != -1:
            snippet = snippet[first : last + 1]
        return json.loads(snippet)

    def _build_plan(self, payload: Dict[str, Any], *, reflection: ReflectionOutput) -> PlannerPlan:
        """Normalise the raw JSON payload into a PlannerPlan."""
        fallback_mode = reflection.mode or "chat"

        mode = str(payload.get("mode", fallback_mode)).strip().lower()
        if mode not in {"chat", "knowledge", "world_query", "world_control"}:
            mode = (
                fallback_mode
                if fallback_mode in {"chat", "knowledge", "world_query", "world_control"}
                else "chat"
            )

        analysis = str(payload.get("analysis", "")).strip()
        steps = [
            str(step).strip()
            for step in (payload.get("steps") or [])
            if str(step).strip()
        ]

        tool_calls = self._parse_tool_calls(payload.get("tool_calls", []))

        response_type = str(payload.get("response_type", "text")).strip().lower()
        if response_type not in {"text", "json"}:
            response_type = "text"

        needs_risk_check = bool(
            payload.get("needs_risk_check", mode in {"world_query", "world_control"})
        )

        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        # ------------------------------------------------------------------
        # Enforce mode-specific expectations
        # ------------------------------------------------------------------

        # In pure knowledge / chat, we never execute tools.
        if mode == "knowledge":
            tool_calls = []
            needs_risk_check = False
        elif mode == "chat":
            tool_calls = []

        # For world_query/world_control, ensure we at least have a safe default.
        goal_text = (reflection.goal or "").lower()

        if mode == "world_query" and not tool_calls:
            if "gpu" in goal_text or "nvidia-smi" in goal_text:
                tool_calls = [ToolCall(name="read_gpu_temps", arguments={})]
            elif "fan" in goal_text:
                tool_calls = [ToolCall(name="read_fan_speeds", arguments={})]
            elif "load" in goal_text or "cpu" in goal_text:
                tool_calls = [ToolCall(name="read_system_load", arguments={})]
            else:
                tool_calls = [ToolCall(name="read_state", arguments={})]

        if mode == "world_control" and not tool_calls:
            # Safe no-op control so the rest of the pipeline can still reason.
            tool_calls = [
                ToolCall(
                    name="noop_control",
                    arguments={"reason": "planner_fallback_noop"},
                )
            ]

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
        """Coerce the 'tool_calls' array into a list[ToolCall]."""
        tool_calls: List[ToolCall] = []
        if not isinstance(raw_calls, list):
            return tool_calls

        for entry in raw_calls:
            if not isinstance(entry, dict):
                continue
            name = str(
                entry.get("name")
                or entry.get("tool_name")
                or entry.get("tool")
                or ""
            ).strip()
            if not name:
                continue
            arguments = entry.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(name=name, arguments=arguments))
        return tool_calls

    # ------------------------------------------------------------------ #
    # Fallback behaviour
    # ------------------------------------------------------------------ #

    def _fallback_plan(self, reflection: ReflectionOutput) -> PlannerPlan:
        """Deterministic fallback if the planner output cannot be parsed."""
        mode = reflection.mode or "chat"
        goal = reflection.goal or "Engage user"
        tool_calls: List[ToolCall] = []

        if mode == "world_query":
            tool_calls = [ToolCall(name="read_state", arguments={})]
        elif mode == "world_control":
            tool_calls = [
                ToolCall(
                    name="noop_control",
                    arguments={"reason": "reflection_fallback_noop"},
                )
            ]

        return PlannerPlan(
            mode=mode,  # type: ignore[arg-type]
            analysis=goal,
            steps=[goal],
            tool_calls=tool_calls,
            response_type="text",
            needs_risk_check=mode in {"world_query", "world_control"},
            confidence=0.0,
        )


__all__ = ["PlannerRunner"]