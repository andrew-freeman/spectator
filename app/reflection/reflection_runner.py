"""Reflection layer that classifies user intent for Spectator V2."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import ReflectionOutput


class SupportsGenerate(Protocol):
    """Subset of the LLM client API required by the reflection layer."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are the REFLECTION layer for Spectator, a local hierarchical agent.
Your job is to classify the user's message into one of four MODES and
produce a concise JSON summary before any planning occurs.

MODES:
- chat: greetings, identity questions, simple acknowledgements.
- knowledge: general knowledge or reasoning that does not need local tools.
- world_query: requests for real system state (GPU temps, sensors, etc.).
- world_control: instructions that change system behaviour (set fan speeds, policies).

Respond with STRICT JSON using this schema:
{{
  "mode": "knowledge",
  "goal": "Concise description of what the user wants",
  "context": {{}},
  "needs_clarification": false,
  "reflection_notes": "Why you chose the mode."
}}

Be deterministic. Do not mention tools by name. Only fill the JSON object.

Example:
{{
  "mode": "knowledge",
  "goal": "Compute 2+2",
  "context": {{}},
  "needs_clarification": false,
  "reflection_notes": "User is asking for a simple arithmetic calculation."
}}

USER MESSAGE:
'''{message}'''
""".strip()


class ReflectionRunner:
    """Generate ReflectionOutput objects from raw user messages."""

    def __init__(self, client: SupportsGenerate, identity_profile: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity_profile or {}

    def run(self, message: str) -> ReflectionOutput:
        prompt = REFLECTION_PROMPT.format(message=message)
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = self._parse_json(raw)
            return self._build_output(payload)
        except Exception:
            return ReflectionOutput(
                mode="chat",
                goal=message.strip() or "General conversation",
                context={},
                needs_clarification=False,
                reflection_notes="Reflection fallback used due to parsing error.",
            )

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        snippet = raw.strip()
        first = snippet.find("{")
        last = snippet.rfind("}")
        if first != -1 and last != -1:
            snippet = snippet[first : last + 1]
        return json.loads(snippet)

    def _build_output(self, payload: Dict[str, Any]) -> ReflectionOutput:
        mode = str(payload.get("mode", "chat")).strip().lower()
        if mode not in {"chat", "knowledge", "world_query", "world_control"}:
            mode = "chat"
        goal = str(payload.get("goal", "")).strip() or "Engage user"
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        needs_clarification = bool(payload.get("needs_clarification", False))
        notes = str(payload.get("reflection_notes", "")).strip()
        return ReflectionOutput(
            mode=mode,  # type: ignore[arg-type]
            goal=goal,
            context=context,
            needs_clarification=needs_clarification,
            reflection_notes=notes,
        )
