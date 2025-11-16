"""Reflection layer that classifies user intent for Spectator V2."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import ReflectionOutput


class SupportsGenerate(Protocol):
    """Subset of the LLM client API required by the reflection layer."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are the REFLECTION layer for Spectator, a local hierarchical agent.
Your job is to classify the user's message BEFORE any planning or tools run.

You MUST output STRICT JSON ONLY, following this schema exactly:
{{
  "mode": "knowledge",
  "goal": "Concise imperative command",
  "context": {{}},
  "needs_clarification": false,
  "reflection_notes": "Short justification"
}}

--------------------------------
MODE CLASSIFICATION RULES
--------------------------------

chat:
- Greetings: "hi", "hello", "are you there"
- Identity questions: "who are you", "what are you"
- Simple acknowledgements, emotional or casual conversation
- goal: short imperative like "Answer identity question" or "Acknowledge presence"

knowledge:
- Pure reasoning or world knowledge
- Math: "2+2", "integrate x dx"
- Logic / physics puzzles (nut in a cup, mirrors, etc.)
- Explanations and definitions
- NO access to real hardware or sensors
- goal: imperative like "Compute 2+2", "Explain quantum computing"

world_query:
- Requests for REAL system state:
  - GPU temperatures, nvidia-smi
  - fan speeds, sensor values
  - system load, controller state
- goal: imperative like "Read GPU temperatures", "Fetch current fan speeds"

world_control:
- Instructions that CHANGE system behaviour:
  - adjust or set fan speeds
  - lower GPU temperatures
  - change performance or power limits
- goal: imperative like "Lower GPU temperatures below 55C"

--------------------------------
STRICT OUTPUT REQUIREMENTS
--------------------------------

- Always choose exactly ONE mode from: "chat", "knowledge", "world_query", "world_control".
- "goal" MUST be an imperative command, NOT a description of what the user wants.
  GOOD: "Compute 2+2"
  BAD: "The user wants to compute 2+2"
- "context" MUST be a JSON object (use {{}} if you have nothing to add).
- "reflection_notes" MUST briefly justify the chosen mode.
- NEVER mention tools by name.
- NEVER include identity strings like "I am Spectator" in any field.
- NEVER output markdown or prose outside the JSON object.

USER MESSAGE:
'''{message}'''
""".strip()


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
        prompt = REFLECTION_PROMPT.format(message=message)
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = self._parse_json(raw)
            output = self._build_output(payload, fallback_message=message)

            # --- Lightweight post-heuristics for stability ---
            lower = message.lower()

            # Simple math / symbolic expressions → knowledge
            #if any(sym in lower for sym in ["2+2", " 2 + 2", "+", "-", "*", "/", "integrate", "derivative"]):
            if re.search(r"\d+\s*[\+\-\*/]\s*\d+", lower):
                if output.mode not in {"world_query", "world_control"}:
                    output.mode = "knowledge"  # type: ignore[attr-defined]

            # Classic reasoning / puzzle keywords → knowledge
            if any(kw in lower for kw in ["cup", "nut inside", "mirror", "reflection puzzle"]):
                if output.mode not in {"world_query", "world_control"}:
                    output.mode = "knowledge"  # type: ignore[attr-defined]

            return output

        except Exception:
            return ReflectionOutput(
                mode="chat",
                goal=message.strip() or "General conversation",
                context={},
                needs_clarification=False,
                reflection_notes="Reflection fallback used due to parsing error.",
            )

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Robustly extract a JSON object from the model output."""
        snippet = raw.strip()
        first = snippet.find("{")
        last = snippet.rfind("}")
        if first != -1 and last != -1:
            snippet = snippet[first : last + 1]
        return json.loads(snippet)

    def _build_output(
        self,
        payload: Dict[str, Any],
        *,
        fallback_message: str,
    ) -> ReflectionOutput:
        """Normalise the raw JSON dict into a ReflectionOutput."""
        mode = str(payload.get("mode", "chat")).strip().lower()
        if mode not in {"chat", "knowledge", "world_query", "world_control"}:
            mode = "chat"

        goal_raw = str(payload.get("goal", "")).strip()
        if not goal_raw:
            # Fallback: derive a simple imperative from the message.
            if mode == "knowledge":
                goal_raw = "Answer the question"
            elif mode == "world_query":
                goal_raw = "Read system state"
            elif mode == "world_control":
                goal_raw = "Adjust system behaviour"
            else:
                goal_raw = fallback_message.strip() or "Engage user"

        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}

        needs_clarification = bool(payload.get("needs_clarification", False))
        #notes = str(payload.get("reflection_notes", "")).strip()
        notes = payload.get("reflection_notes") or ""

        return ReflectionOutput(
            mode=mode,  # type: ignore[arg-type]
            goal=goal_raw,
            context=context,
            needs_clarification=needs_clarification,
            reflection_notes=notes,
        )


__all__ = ["ReflectionRunner"]