"""Critic runner that validates planner outputs for safety and consistency."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Protocol

from app.core.schemas import CriticOutput, Plan, ToolCall

from .critic_prompt import build_critic_prompt


class SupportsGenerate(Protocol):
    """Protocol representing the shared language model interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


class CriticRunner:
    """Prepare prompt, call model, and parse structured critic feedback."""

    def __init__(self, client: SupportsGenerate, *, identity: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    def run(self, plan: Plan, safety_policies: Optional[Iterable[str]] = None) -> CriticOutput:
        plan_payload = _plan_to_payload(plan)
        prompt = build_critic_prompt(
            plan_payload,
            safety_policies=list(safety_policies or []),
            identity=self._identity,
            policy=self._policy,
        )
        try:
            raw = self._client.generate(prompt, stop=None)
            payload = _parse_json(raw)
        except Exception:
            return CriticOutput(
                risk="low",
                issues=[],
                suggestions=[],
                confidence=0.0,
                notes="Critic fallback invoked; assuming low risk.",
            )

        risk = str(payload.get("risk", "low")).strip().lower()
        if risk not in {"low", "medium", "high", "unsafe"}:
            risk = "low"
        issues = [str(i).strip() for i in payload.get("issues", []) if str(i).strip()]
        suggestions = [
            str(s).strip() for s in payload.get("suggestions", []) if str(s).strip()
        ]
        adjusted_steps = [
            str(s).strip() for s in payload.get("adjusted_steps", []) if str(s).strip()
        ]
        adjusted_tool_calls = []
        for tc in payload.get("adjusted_tool_calls", []):
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get("tool_name") or tc.get("name") or ""
            adjusted_tool_calls.append(
                ToolCall(tool_name=tool_name, arguments=tc.get("arguments", {}) or {})
            )
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        notes = str(payload.get("notes", "")).strip()

        return CriticOutput(
            risk=risk,  # type: ignore[arg-type]
            issues=issues,
            suggestions=suggestions,
            adjusted_steps=adjusted_steps,
            adjusted_tool_calls=adjusted_tool_calls,
            confidence=confidence,
            notes=notes,
        )


def _plan_to_payload(plan: Plan) -> Dict[str, Any]:
    return {
        "analysis": plan.analysis,
        "steps": plan.steps,
        "tool_calls": [
            {"tool_name": tc.tool_name, "arguments": tc.arguments}
            for tc in plan.tool_calls
        ],
        "response_type": plan.response_type,
        "needs_risk_check": plan.needs_risk_check,
        "confidence": plan.confidence,
    }


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Critic returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["CriticRunner"]
