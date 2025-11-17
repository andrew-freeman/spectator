# app/reflection/reflection_runner_v3.py
"""Reflection V3 runner: classify user message into mode + goal + context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.core.schemas import Mode, ReflectionOutput

from .reflection_prompt_v3 import REFLECTION_PROMPT_V3

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Protocol for the LLM client used by the reflection module."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class ReflectionSections:
    mode: str
    goal: str
    needs_clarification: bool
    context: Dict[str, Any]
    notes: str


class ReflectionRunnerV3:
    """Call the LLM reflection module and normalise its output into ReflectionOutput."""

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        identity_profile: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._identity = identity_profile or {}
        self._policy = policy or {}

    # Public API (same shape as existing ReflectionRunner)
    # ------------------------------------------------------------------
    def run(self, user_message: str) -> ReflectionOutput:
        """Run the V3 reflection prompt and return a normalised ReflectionOutput."""
        prompt = self._build_prompt(user_message)
        try:
            raw = self._client.generate(prompt, stop=None)
            sections = self._parse_sections(raw)
            return self._to_reflection_output(sections, user_message=user_message)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Reflection V3 fallback due to error: %s", exc)
            return self._fallback_reflection(user_message)

    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self, user_message: str) -> str:
        identity_json = json.dumps(self._identity, ensure_ascii=False, indent=2)
        policy_json = json.dumps(self._policy, ensure_ascii=False, indent=2)

        prompt = REFLECTION_PROMPT_V3
        prompt = prompt.replace("<<USER_MESSAGE>>", user_message)
        prompt = prompt.replace("<<IDENTITY_JSON>>", identity_json)
        prompt = prompt.replace("<<POLICY_JSON>>", policy_json)
        return prompt

    # Parsing
    # ------------------------------------------------------------------
    def _parse_sections(self, raw: str) -> ReflectionSections:
        """
        Parse the section-based reflection output.

        Expected headers:
        - #MODE:
        - #GOAL:
        - #NEEDS_CLARIFICATION:
        - #CONTEXT:
        - #NOTES:
        """
        text = raw.strip()
        sections: Dict[str, List[str]] = {
            "MODE": [],
            "GOAL": [],
            "NEEDS_CLARIFICATION": [],
            "CONTEXT": [],
            "NOTES": [],
        }

        current_key: Optional[str] = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#MODE:"):
                current_key = "MODE"
                continue
            if stripped.startswith("#GOAL:"):
                current_key = "GOAL"
                continue
            if stripped.startswith("#NEEDS_CLARIFICATION:"):
                current_key = "NEEDS_CLARIFICATION"
                continue
            if stripped.startswith("#CONTEXT:"):
                current_key = "CONTEXT"
                continue
            if stripped.startswith("#NOTES:"):
                current_key = "NOTES"
                continue

            if current_key and current_key in sections:
                sections[current_key].append(stripped)

        # MODE
        mode_raw = " ".join(sections["MODE"]).strip().lower()
        mode: Mode
        if mode_raw in {"chat", "knowledge", "world_query", "world_control"}:
            mode = mode_raw  # type: ignore[assignment]
        else:
            # Simple heuristics as fallback
            lower_msg = " ".join(sections["GOAL"]).lower()
            if any(k in lower_msg for k in ["temp", "gpu", "nvidia-smi", "memory usage", "utilization"]):
                mode = "world_query"  # type: ignore[assignment]
            elif any(k in lower_msg for k in ["fan", "power limit", "keep under", "increase speed"]):
                mode = "world_control"  # type: ignore[assignment]
            elif "who are you" in lower_msg or "what are you" in lower_msg:
                mode = "chat"  # type: ignore[assignment]
            else:
                mode = "knowledge"  # type: ignore[assignment]

        # GOAL
        goal = " ".join(sections["GOAL"]).strip()
        if not goal:
            goal = "Interpret and respond to the user message."

        # NEEDS_CLARIFICATION
        needs_raw = " ".join(sections["NEEDS_CLARIFICATION"]).strip().lower()
        needs_clarification = needs_raw == "true"

        # CONTEXT
        context: Dict[str, Any] = {}
        for line in sections["CONTEXT"]:
            # Expect "- key: value" or "key: value"
            cleaned = line.lstrip("-").strip()
            if not cleaned:
                continue
            if ":" in cleaned:
                key, value = cleaned.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    context[key] = value
            else:
                # Fallback: append anonymous flags as numbered items
                idx = len(context)
                context[f"item_{idx}"] = cleaned

        # NOTES
        notes = " ".join(sections["NOTES"]).strip()

        return ReflectionSections(
            mode=mode,
            goal=goal,
            needs_clarification=needs_clarification,
            context=context,
            notes=notes,
        )

    # Conversion and fallback
    # ------------------------------------------------------------------
    def _to_reflection_output(self, sections: ReflectionSections, *, user_message: str) -> ReflectionOutput:
        # Merge the parsed context with simple top-level flags
        context = dict(sections.context)

        # Example flags we may want to carry downstream:
        # - chat_mode (for responder)
        # - goal_update / objective flags could be added later
        if sections.mode == "chat":
            context.setdefault("chat_mode", True)

        return ReflectionOutput(
            mode=sections.mode,  # type: ignore[arg-type]
            goal=sections.goal or user_message,
            context=context,
            needs_clarification=sections.needs_clarification,
            reflection_notes=sections.notes,
        )

    def _fallback_reflection(self, user_message: str) -> ReflectionOutput:
        """Deterministic fallback if the reflection output cannot be parsed."""
        return ReflectionOutput(
            mode="knowledge",
            goal=user_message or "Engage with the user query.",
            context={},
            needs_clarification=False,
            reflection_notes="Reflection V3 fallback due to parsing or runtime error.",
        )


__all__ = ["ReflectionRunnerV3"]