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
You are a careful reflection module that analyses a user message before any tools run.
For the provided USER MESSAGE, respond strictly as a JSON object with the following keys:
- intent: one of ["query", "command", "objective", "ambiguous"].
- refined_objectives: array of concise objective strings (may be empty).
- context: JSON object of contextual hints or structured slots (may be empty).
- needs_clarification: boolean indicating whether you must ask for clarification.
- reflection_notes: short natural-language explanation summarising your reasoning.

Example format:
{
  "intent": "command",
  "refined_objectives": ["Deploy fixes to API"],
  "context": {"priority": "high"},
  "needs_clarification": false,
  "reflection_notes": "User requested a deployment."
}

USER MESSAGE:
{message}
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
        if intent not in {"query", "command", "objective", "ambiguous"}:
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
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = json.loads(raw)
            output = ReflectionOutput.from_payload(payload)
        except Exception:
            # Defensive fallback keeps pipeline resilient.
            output = ReflectionOutput(
                intent="ambiguous",
                refined_objectives=[],
                context={},
                needs_clarification=False,
                reflection_notes="Reflection fallback invoked; defaulting to standard planning.",
            )
        return output.to_dict()


__all__ = ["ReflectionRunner", "ReflectionOutput"]
