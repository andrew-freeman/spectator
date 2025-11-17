# app/critic/critic_runner.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from app.core.schemas import CriticOutput, PlannerPlan, ReflectionOutput, ToolResult

from .critic_prompt_builder_v3 import build_critic_prompt_v3
from .critic_output_parser_v3 import CriticParseError, parse_critic_output_v3

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Protocol for the LLM client used by the critic."""
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class CriticRunner:
    """
    V3 Critic Runner
    - Builds strict section-based critic prompt
    - Parses section-based critic output
    - If parsing fails, performs a repair pass
    - If the repair pass also fails → fallback conservative evaluation
    """

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        identity: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #

    def run(
        self,
        reflection: ReflectionOutput,
        plan: PlannerPlan,
        tool_results: Sequence[ToolResult],
        current_state: Dict[str, Any],
    ) -> CriticOutput:
        """Evaluate the proposed plan and tool calls for safety and risk."""

        base_prompt = build_critic_prompt_v3(
            reflection,
            plan,
            tool_results,
            current_state or {},
            identity=self._identity,
            policy=self._policy,
        )

        raw = self._client.generate(base_prompt, stop=None)
        try:
            payload = parse_critic_output_v3(raw)
            return self._build_output(payload)
        except CriticParseError as exc:
            LOGGER.warning("Critic V3 parse failed: %s", exc)

        repair_prompt = self._build_repair_prompt(raw)
        repaired_raw = self._client.generate(repair_prompt, stop=None)

        try:
            payload = parse_critic_output_v3(repaired_raw)
            return self._build_output(payload)
        except CriticParseError as exc:
            LOGGER.error("Critic repair pass also failed: %s", exc)

        # Fallback: conservative but not catastrophic
        return self._fallback_output()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_output(self, payload: Dict[str, Any]) -> CriticOutput:
        risk_level = str(payload.get("risk_level", "medium")).lower()
        if risk_level not in {"low", "medium", "high", "unsafe"}:
            risk_level = "medium"

        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        issues_raw = payload.get("detected_issues") or []
        detected_issues: List[str] = [
            str(issue).strip() for issue in issues_raw if str(issue).strip()
        ]

        notes = str(payload.get("notes", "")).strip()

        return CriticOutput(
            risk_level=risk_level,  # type: ignore[arg-type]
            confidence=confidence,
            detected_issues=detected_issues,
            notes=notes,
        )

    def _build_repair_prompt(self, bad_output: str) -> str:
        return f"""
Your previous output did not follow the required Spectator V3 CRITIC format.

Here is your invalid output:

{bad_output}

Please output a corrected version using EXACTLY these sections:

#RISK_LEVEL:
#CONFIDENCE:
#ISSUES:
#NOTES:

Rules:
- Do not output JSON.
- Do not output markdown.
- Do not add commentary outside the section headers.
- Keep your risk judgment the same; fix only the formatting.
"""

    def _fallback_output(self) -> CriticOutput:
        """
        Deterministic fallback if the critic output cannot be parsed.
        Default to a conservative 'medium' risk with low confidence.
        """
        return CriticOutput(
            risk_level="medium",
            confidence=0.0,
            detected_issues=["Failed to parse critic output; defaulting to medium risk."],
            notes="The critic could not be parsed, so a conservative default was applied.",
        )


__all__ = ["CriticRunner"]