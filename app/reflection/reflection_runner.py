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
You are a careful reflection module that analyses a user message before any tools run.
For the provided USER MESSAGE, respond strictly as a JSON object with the following keys:
- intent: one of ["query", "command", "objective", "ambiguous"].
- query_type: one of ["world", "knowledge"].
- refined_objectives: array of concise objective strings (may be empty).
- context: JSON object of contextual hints or structured slots (may be empty). Use "chat_mode": true when the user is
  purely conversing.
- needs_clarification: boolean.
- reflection_notes: short natural-language explanation.

IDENTITY PROFILE:
{identity_block}

Example:
{{
  "intent": "query",
  "query_type": "world",
  "refined_objectives": ["Retrieve GPU readings"],
  "context": {{
    "query_mode": true
  }},
  "needs_clarification": false,
  "reflection_notes": "User is requesting information."
}}

USER MESSAGE:
'''{message}'''
"""
).strip()


@dataclass
class ReflectionOutput:
    intent: str = "ambiguous"
    query_type: str = "knowledge"
    refined_objectives: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    reflection_notes: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReflectionOutput":
        intent = str(payload.get("intent", "ambiguous")).strip().lower() or "ambiguous"
        if intent not in {"query", "command", "objective", "chat", "ambiguous"}:
            intent = "ambiguous"
        query_type = str(payload.get("query_type", "knowledge")).strip().lower() or "knowledge"
        if query_type not in {"world", "knowledge"}:
            query_type = "knowledge"
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
            query_type=query_type,
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
            self._assign_query_type(output, message)
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
            self._assign_query_type(fallback, message)
            return fallback.to_dict()

        try:
            payload = json.loads(raw)
            output = ReflectionOutput.from_payload(payload)
            self._apply_intent_context(output)
            self._assign_query_type(output, message)
            return output.to_dict()
        except Exception:
            self._assign_query_type(fallback, message)
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

    def _assign_query_type(self, output: ReflectionOutput, message: str) -> None:
        query_type = self._determine_query_type(message)
        output.query_type = query_type
        ctx = output.context
        if not isinstance(ctx, dict):
            ctx = {}
            output.context = ctx
        ctx["query_type"] = query_type

    def _determine_query_type(self, message: str) -> str:
        text = (message or "").strip().lower()
        if not text:
            return "knowledge"
        world_tokens = (
            "temperature",
            "temps",
            "gpu",
            "nvidia-smi",
            "fan",
            "rpm",
            "load",
            "usage",
            "utilization",
        )
        if any(token in text for token in world_tokens):
            return "world"
        math_like = bool(re.search(r"\d+\s*[+\-*/]", message or ""))
        identity_like = any(
            phrase in text
            for phrase in (
                "who are you",
                "what is your name",
                "what is 2+2",
                "define",
                "explain",
            )
        )
        if math_like or identity_like:
            return "knowledge"
        return "knowledge"

    def _is_simple_query(self, message: str) -> bool:
        text = (message or "").strip()
        if not text:
            return False
        if self._is_identity_question(text):
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

    def _is_identity_question(self, text: str) -> bool:
        lowered = text.lower()
        return "who are you" in lowered or lowered.strip() in {"who r u", "who ru"}


__all__ = ["ReflectionRunner", "ReflectionOutput"]
