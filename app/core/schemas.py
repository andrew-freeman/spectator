"""Pydantic models used across the Spectator V3 pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["chat", "knowledge", "world_query", "world_control"]


class ToolCall(BaseModel):
    """Structured representation of a tool invocation."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        return value.strip()


class ToolResult(BaseModel):
    """Result returned by the ToolExecutor for a ToolCall."""

    tool: str
    status: Literal["ok", "error"]
    result: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ReflectionOutput(BaseModel):
    """Pure intent classification emitted by the reflection layer."""

    mode: Mode
    goal: str
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("goal")
    @classmethod
    def _clean_goal(cls, value: str) -> str:
        text = value.strip()
        return text or "Engage user"


class PlannerPlan(BaseModel):
    """Structured plan describing the steps and tools selected by the planner."""

    mode: Mode
    analysis: str
    steps: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    response_type: Literal["text", "json"] = "text"
    needs_risk_check: bool = False
    confidence: float = 0.0


class CriticReview(BaseModel):
    """Risk assessment returned by the critic."""

    risk_level: Literal["low", "medium", "high", "unsafe"] = "low"
    confidence: float = 0.0
    detected_issues: List[str] = Field(default_factory=list)
    notes: str = ""


class GovernorDecision(BaseModel):
    """Final arbitration outcome that dictates execution and response."""

    verdict: str
    rationale: str
    final_tool_calls: List[ToolCall] = Field(default_factory=list)
    response_type: Literal["text", "json"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResponderFrame(BaseModel):
    """Final response artifact handed to the chat UI."""

    final_text: str
    short_summary: str
    used_tools: List[str] = Field(default_factory=list)
    mode: Mode


__all__ = [
    "Mode",
    "ToolCall",
    "ToolResult",
    "ReflectionOutput",
    "PlannerPlan",
    "CriticReview",
    "GovernorDecision",
    "ResponderFrame",
]
