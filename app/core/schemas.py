"""Shared dataclasses for the Spectator v2 pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

Mode = Literal["chat", "knowledge", "world_query", "world_control", "ambiguous"]


@dataclass
class ReflectionOutput:
    mode: Mode
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    reflection_notes: str = ""
    original_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_name(self) -> str:
        """Backwards compatible accessor for older components."""
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class PlannerPlan:
    analysis: str
    steps: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    response_type: Literal["text", "json"] = "text"
    needs_risk_check: bool = True
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "steps": list(self.steps),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "response_type": self.response_type,
            "needs_risk_check": self.needs_risk_check,
            "confidence": self.confidence,
        }


@dataclass
class CriticOutput:
    risk: Literal["low", "medium", "high", "unsafe"] = "low"
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    adjusted_steps: List[str] = field(default_factory=list)
    adjusted_tool_calls: List[ToolCall] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk,
            "issues": list(self.issues),
            "suggestions": list(self.suggestions),
            "adjusted_steps": list(self.adjusted_steps),
            "adjusted_tool_calls": [call.to_dict() for call in self.adjusted_tool_calls],
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class GovernorDecision:
    verdict: str
    rationale: str = ""
    final_tool_calls: List[ToolCall] = field(default_factory=list)
    final_response_type: Literal["text", "json"] = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "final_tool_calls": [call.to_dict() for call in self.final_tool_calls],
            "final_response_type": self.final_response_type,
            "metadata": dict(self.metadata),
        }


@dataclass
class ToolResult:
    tool: str
    status: Literal["ok", "error"] = "ok"
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


__all__ = [
    "Mode",
    "ReflectionOutput",
    "ToolCall",
    "PlannerPlan",
    "CriticOutput",
    "GovernorDecision",
    "ToolResult",
]
