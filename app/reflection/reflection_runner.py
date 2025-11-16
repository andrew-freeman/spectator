"""Reflection runner responsible for pre-processing user intent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol


class SupportsGenerate(Protocol):
    """Protocol describing the shared LLM client interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are a careful reflection module that analyses the USER MESSAGE before any tools run.

Return STRICT JSON with these keys:
- intent: one of ["query", "command", "objective", "chat", "ambiguous"]
- refined_objectives: array of concise objectives (may be empty)
- context: JSON object (may be empty)
- needs_clarification: boolean
- reflection_notes: short explanation

Classification rules:
- If user asks about identity, opinions, personality → intent="chat"
- If user engages in casual conversation → intent="chat"
- If user asks for system information or readings → intent="query"
- If user asks to change system state or apply settings → intent="command"
- If user sets a high-level goal → intent="objective"

Example output:
{{
  "intent": "chat",
  "refined_objectives": [],
  "context": {{"chat_mode": true}},
  "needs_clarification": false,
  "reflection_notes": "User is asking about the agent's identity."
}}

USER MESSAGE:
'''{message}'''
""".strip()


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

    def __init__(self, client: SupportsGenerate):
        self._client = client

    def run(self, message: str) -> Dict[str, Any]:
        prompt = REFLECTION_PROMPT.format(message=message)
        fallback = ReflectionOutput(
            intent="ambiguous",
            refined_objectives=[],
            context={},
            needs_clarification=False,
            reflection_notes="Reflection fallback invoked.",
        )

        try:
            raw = self._client.generate(prompt, stop=None)
            print("RAW REFLECTION OUTPUT:", raw)
        except Exception:
            return fallback.to_dict()

        try:
            payload = json.loads(raw)
            output = ReflectionOutput.from_payload(payload)
            if output.intent == "chat":
                output.context["chat_mode"] = True
            return output.to_dict()
        except Exception:
            return fallback.to_dict()


__all__ = ["ReflectionRunner", "ReflectionOutput"]
