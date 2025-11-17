"""Shared dataclasses used throughout the Spectator v2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Mode = Literal["chat", "knowledge", "world_query", "world_control"]


@dataclass
class ToolCall:
    """Structured representation of a tool invocation."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass
class ToolResult:
    """Result returned by the ToolExecutor for a ToolCall."""

    tool: str
    status: Literal["ok", "error"]
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "tool": self.tool,
            "status": self.status,
            "result": dict(self.result),
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class ReflectionOutput:
    """Normalized intent classification emitted by the reflection layer."""

    mode: Mode
    goal: str
    context: Dict[str, Any]
    needs_clarification: bool
    reflection_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "goal": self.goal,
            "context": dict(self.context),
            "needs_clarification": self.needs_clarification,
            "reflection_notes": self.reflection_notes,
        }


@dataclass
class PlannerPlan:
    """Structured plan describing the steps and tools selected by the planner."""

    mode: Mode
    analysis: str
    steps: List[str]
    tool_calls: List[ToolCall]
    response_type: Literal["text", "json"]
    needs_risk_check: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "analysis": self.analysis,
            "steps": list(self.steps),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "response_type": self.response_type,
            "needs_risk_check": self.needs_risk_check,
            "confidence": self.confidence,
        }


@dataclass
class CriticOutput:
    """Risk assessment returned by the critic."""

    risk_level: Literal["low", "medium", "high", "unsafe"]
    confidence: float
    detected_issues: List[str]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "detected_issues": list(self.detected_issues),
            "notes": self.notes,
        }


@dataclass
class GovernorDecision:
    verdict: str
    rationale: str
    final_tool_calls: List[ToolCall]
    response_type: Literal["text", "json"] = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "final_tool_calls": [call.to_dict() for call in self.final_tool_calls],
            "response_type": self.response_type,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "Mode",
    "ToolCall",
    "ToolResult",
    "ReflectionOutput",
    "PlannerPlan",
    "CriticOutput",
    "GovernorDecision",
]
