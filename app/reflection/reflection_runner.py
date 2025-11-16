"""Reflection runner responsible for pre-processing user intent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol


class SupportsGenerate(Protocol):
    """Protocol describing the shared LLM client interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = (
    """
You are Spectator's reflection analyst. Before any tools run, classify the
incoming USER MESSAGE and set safe defaults.

IDENTITY PROFILE:
{identity_block}

Return STRICT JSON with these keys:
- intent: one of ["query", "command", "objective", "chat", "ambiguous"]
- refined_objectives: array of concise objectives (may be empty)
- context: JSON object (may be empty)
- needs_clarification: boolean
- reflection_notes: short explanation

Classification rules:
- Identity, personality, or "Who are you?" → intent="chat"
- Casual conversation → intent="chat"
- Requests for telemetry or status → intent="query"
- Requests to change settings → intent="command"
- Strategic or multi-cycle goals → intent="objective"

Example output:
{{
  "intent": "chat",
  "refined_objectives": [],
  "context": {{"chat_mode": true, "allowed_tool_kinds": []}},
  "needs_clarification": false,
  "reflection_notes": "User is asking about the agent's identity."
}}

USER MESSAGE:
'''{message}'''
"""
).strip()


@dataclass
class ReflectionOutput:
    intent: str = "ambiguous"
    refined_objectives: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    reflection_notes: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReflectionOutput":
        intent = str(payload.get("intent", "ambiguous")).strip().lower() or "ambiguous"
        if intent not in {"query", "command", "objective", "chat", "ambiguous"}:
            intent = "ambiguous"
        refined_objectives = [
            str(item).strip()
            for item in payload.get("refined_objectives", [])
            if str(item).strip()
        ]
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        needs_clarification = bool(payload.get("needs_clarification", False))
        reflection_notes = str(payload.get("reflection_notes", "")).strip()
        return cls(
            intent=intent,
            refined_objectives=refined_objectives,
            context=context,
            needs_clarification=needs_clarification,
            reflection_notes=reflection_notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReflectionRunner:
    """Generate a structured reflection summary prior to full reasoning."""

    def __init__(self, client: SupportsGenerate, *, identity_profile: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity_profile = identity_profile or {}

    def run(self, message: str) -> Dict[str, Any]:
        if self._is_simple_query(message):
            output = ReflectionOutput(
                intent="query",
                refined_objectives=[],
                context={"query_mode": True, "allowed_tool_kinds": ["sensor"]},
                needs_clarification=False,
                reflection_notes="Simple query detected; bypassing reflection model.",
            )
            return output.to_dict()

        prompt = REFLECTION_PROMPT.format(
            message=message,
            identity_block=json.dumps(self._identity_profile, indent=2),
        )
        fallback = ReflectionOutput(
            intent="ambiguous",
            refined_objectives=[],
            context={},
            needs_clarification=False,
            reflection_notes="Reflection fallback invoked.",
        )

        try:
            raw = self._client.generate(prompt, stop=None)
        except Exception:
            return fallback.to_dict()

        try:
            payload = json.loads(raw)
            output = ReflectionOutput.from_payload(payload)
            self._apply_intent_context(output)
            return output.to_dict()
        except Exception:
            return fallback.to_dict()

    def _apply_intent_context(self, output: ReflectionOutput) -> None:
        ctx = output.context
        if output.intent == "chat":
            ctx["chat_mode"] = True
            ctx["allowed_tool_kinds"] = []
        elif output.intent == "query":
            ctx["query_mode"] = True
            ctx["allowed_tool_kinds"] = ["sensor"]
        elif output.intent == "command":
            ctx["command_mode"] = True
            ctx["allowed_tool_kinds"] = ["sensor", "actuator"]
            ctx["force_action"] = True
        elif output.intent == "objective":
            ctx["goal_update"] = True

    def _is_simple_query(self, message: str) -> bool:
        text = (message or "").strip()
        if not text:
            return False
        return self._is_simple_math(text) or self._is_simple_natural_question(text)

    def _is_simple_math(self, text: str) -> bool:
        return bool(re.fullmatch(r"[0-9\s+\-*/^().=]+", text))

    def _is_simple_natural_question(self, text: str) -> bool:
        lowered = text.lower()
        if len(lowered) > 60:
            return False
        if len(lowered.split()) > 12:
            return False
        question_words = ("who", "what", "where", "when", "why", "how")
        return any(
            lowered.startswith(word)
            or lowered.startswith(f"{word} ")
            or lowered.startswith(f"{word}'s")
            for word in question_words
        )


__all__ = ["ReflectionRunner", "ReflectionOutput"]
