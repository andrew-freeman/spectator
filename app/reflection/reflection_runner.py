"""Reflection layer that classifies user intent for Spectator V3."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import ReflectionOutput
from app.core.structured import assemble_prompt, generate_structured_object


class SupportsGenerate(Protocol):
    """Subset of the LLM client API required by the reflection layer."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = "\n".join(
    [
        "Role: Spectator reflection classifier.",
        "Constraints:",
        "- Emit JSON content with keys: mode, goal, context.",
        "- Mode must be one of chat, knowledge, world_query, world_control.",
        "- Goal must be an imperative statement under twelve words.",
        "- Context must be a JSON object (use an empty object when nothing applies).",
        "- Never mention tools or Spectator's identity.",
        "User message:",
        "{message}",
        "Respond with JSON content only (no braces).",
    ]
)


class ReflectionRunner:
    """Generate ReflectionOutput objects from raw user messages."""

    def __init__(
        self,
        client: SupportsGenerate,
        identity_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._identity = identity_profile or {}

    def run(self, message: str) -> ReflectionOutput:
        prompt = assemble_prompt(
            REFLECTION_PROMPT.format(message=message.strip()),
            "Identity context is available for grounding only; never echo it.",
            str(self._identity),
        )

        def _fallback(_: Exception | None = None) -> ReflectionOutput:
            return ReflectionOutput(mode="chat", goal=message.strip() or "Engage user", context={})

        output = generate_structured_object(self._client, prompt, ReflectionOutput, _fallback)

        lower = message.lower()
        if self._looks_like_math(lower) and output.mode not in {"world_query", "world_control"}:
            output = output.model_copy(update={"mode": "knowledge"})

        if any(kw in lower for kw in ["cup", "mirror", "riddle", "nut"]):
            if output.mode not in {"world_query", "world_control"}:
                output = output.model_copy(update={"mode": "knowledge"})

        if not output.goal.strip():
            default_goal = {
                "chat": "Answer briefly",
                "knowledge": "Explain the topic",
                "world_query": "Read system state",
                "world_control": "Adjust system safely",
            }[output.mode]
            output = output.model_copy(update={"goal": default_goal})

        return output

    @staticmethod
    def _looks_like_math(text: str) -> bool:
        symbols = ["+", "-", "*", "/", "integrate", "derivative", "sqrt"]
        return any(sym in text for sym in symbols)


__all__ = ["ReflectionRunner", "REFLECTION_PROMPT"]
