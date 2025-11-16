"""Reflection runner responsible for classifying and rewriting user intent."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import ReflectionOutput

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Protocol describing the shared LLM client interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are the reflection layer for Spectator, a local workstation agent.
Your task is to interpret the USER MESSAGE before any planning occurs.
Respond with **STRICT JSON ONLY** following this schema:
{{
  "mode": "chat | knowledge | world_query | world_control",
  "goal": "concise restatement of the user intent",
  "context": {{"key": "value"}}  // include "chat_mode": true for conversation requests
  "needs_clarification": false,
  "reflection_notes": "short reasoning",
  "metadata": {{}}
}}

Legacy compatibility: mention "intent" in your reflection_notes summary to help downstream tooling.

The mode descriptions:
- "chat": general conversation or persona questions.
- "knowledge": questions that can be answered without tools.
- "world_query": questions about the real system state that require tools.
- "world_control": requests to change the system (fan speeds, policies, etc.).

Example response:
{{
  "mode": "world_query",
  "goal": "Retrieve current GPU temperatures",
  "context": {{"query_target": "gpu_temps"}},
  "needs_clarification": false,
  "reflection_notes": "User explicitly asked for nvidia-smi readings.",
  "metadata": {{"requires_tools": true}}
}}

USER MESSAGE:
\"\"\"{message}\"\"\"

IDENTITY CONTEXT:
{identity_block}
""".strip()


class ReflectionRunner:
    """Generate a structured reflection summary prior to full reasoning."""

    def __init__(self, client: SupportsGenerate, *, identity_profile: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity_profile = identity_profile or {}

    def run(self, message: str) -> ReflectionOutput:
        normalized_message = str(message or "").strip()
        prompt = REFLECTION_PROMPT.format(
            message=normalized_message,
            identity_block=json.dumps(self._identity_profile, indent=2, ensure_ascii=False),
        )
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = json.loads(raw)
            return self._to_output(payload, normalized_message)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Reflection fallback triggered: %s", exc)
            return ReflectionOutput(
                mode="knowledge",
                goal=normalized_message or "General inquiry",
                context={},
                needs_clarification=False,
                reflection_notes="Fallback reflection; using raw user message as goal.",
                original_message=normalized_message,
            )

    def _to_output(self, payload: Dict[str, Any], original_message: str) -> ReflectionOutput:
        mode = str(payload.get("mode", "knowledge")).strip().lower() or "knowledge"
        if not payload.get("mode") and payload.get("intent"):
            legacy_map = {
                "chat": "chat",
                "query": "world_query",
                "command": "world_control",
                "objective": "world_control",
            }
            mapped = legacy_map.get(str(payload.get("intent")).strip().lower())
            if mapped:
                mode = mapped
        if mode not in {"chat", "knowledge", "world_query", "world_control"}:
            mode = "knowledge"

        goal = str(payload.get("goal") or original_message or "General inquiry").strip()
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        needs_clarification = bool(payload.get("needs_clarification", False))
        notes = str(payload.get("reflection_notes") or payload.get("notes") or "").strip()
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if mode == "chat":
            context.setdefault("chat_mode", True)
        elif mode == "world_query":
            context.setdefault("query_mode", True)

        output = ReflectionOutput(
            mode=mode,  # type: ignore[arg-type]
            goal=goal,
            context=context,
            needs_clarification=needs_clarification,
            reflection_notes=notes,
            original_message=original_message,
            metadata=metadata,
        )
        LOGGER.info("Reflection classified mode=%s goal=%s", output.mode, output.goal)
        return output


__all__ = ["ReflectionRunner", "SupportsGenerate", "REFLECTION_PROMPT"]
