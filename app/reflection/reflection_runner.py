"""Reflection runner responsible for pre-processing user intent."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import PreprocessorOutput, UserInput


class SupportsGenerate(Protocol):
    """Protocol describing the shared LLM client interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are a careful reflection module that analyses a USER MESSAGE before any tools run.

IDENTITY CONTEXT:
{identity_block}

For the provided USER MESSAGE, respond STRICTLY as a JSON object with the following keys:
- mode: one of ["chat", "knowledge", "world_query", "world_control", "ambiguous"].
- goal: a concise restatement of what the user wants.
- keywords: array of important terms (may be empty).
- requires_tools: boolean indicating whether tools are needed to satisfy the request.
- needs_clarification: boolean.
- clarification_question: a short clarification question IF needs_clarification is true, else null.
- memory_context: array of short strings summarising relevant past events (may be empty).
- context: JSON object of contextual hints (may be empty). Use "chat_mode": true for pure conversation.
- confidence: number in [0,1].
- notes: short natural-language explanation.
- Legacy compatibility: mention "intent" inside your notes when summarising the classification.

Example:
{{
  "mode": "world_query",
  "goal": "Retrieve current GPU temperatures",
  "keywords": ["gpu", "temperature", "nvidia-smi"],
  "requires_tools": true,
  "needs_clarification": false,
  "clarification_question": null,
  "memory_context": [],
  "context": {{"query_mode": true}},
  "confidence": 0.95,
  "notes": "User explicitly asked for readings from nvidia-smi."
}}

USER MESSAGE:
'''{message}'''
""".strip()


class ReflectionRunner:
    """Generate a structured reflection summary prior to full reasoning."""

    def __init__(self, client: SupportsGenerate, *, identity_profile: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity_profile = identity_profile or {}

    def run(self, user_input: UserInput | str) -> PreprocessorOutput | Dict[str, Any]:
        expects_dataclass = isinstance(user_input, UserInput)
        normalized_input = user_input if expects_dataclass else UserInput(raw_text=str(user_input))
        prompt = REFLECTION_PROMPT.format(
            message=normalized_input.raw_text,
            identity_block=json.dumps(self._identity_profile, indent=2),
        )
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = json.loads(raw)
        except Exception:
            fallback = PreprocessorOutput(
                mode="ambiguous",
                goal=normalized_input.raw_text.strip() or "No goal",
                notes="Reflection fallback invoked; defaulting to ambiguous mode.",
            )
            return fallback if expects_dataclass else asdict(fallback)

        legacy_intent = payload.get("intent")
        mode = str(payload.get("mode", "ambiguous")).strip().lower()
        if not payload.get("mode") and legacy_intent:
            intent_map = {
                "chat": "chat",
                "query": "world_query",
                "command": "world_control",
                "objective": "world_control",
            }
            mapped = intent_map.get(str(legacy_intent).lower().strip())
            if mapped:
                mode = mapped
        if mode not in {"chat", "knowledge", "world_query", "world_control", "ambiguous"}:
            mode = "ambiguous"

        goal = str(payload.get("goal", normalized_input.raw_text)).strip() or normalized_input.raw_text
        keywords = [str(k).strip() for k in payload.get("keywords", []) if str(k).strip()]
        requires_tools = bool(payload.get("requires_tools", False))
        needs_clarification = bool(payload.get("needs_clarification", False))
        clarification_question = payload.get("clarification_question")
        if clarification_question is not None:
            clarification_question = str(clarification_question).strip() or None

        memory_context = [
            str(m).strip() for m in payload.get("memory_context", []) if str(m).strip()
        ]
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        if mode == "chat":
            context.setdefault("chat_mode", True)
        elif mode == "world_query":
            context.setdefault("query_mode", True)

        confidence = float(payload.get("confidence", 0.0) or 0.0)
        notes = str(payload.get("notes", "")).strip()

        output = PreprocessorOutput(
            mode=mode,  # type: ignore[arg-type]
            goal=goal,
            keywords=keywords,
            requires_tools=requires_tools,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            memory_context=memory_context,
            context=context,
            confidence=confidence,
            notes=notes,
        )
        return output if expects_dataclass else asdict(output)


__all__ = ["ReflectionRunner", "SupportsGenerate", "REFLECTION_PROMPT"]
