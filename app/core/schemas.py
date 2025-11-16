"""Shared dataclasses for Spectator v2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Mode = Literal["chat", "knowledge", "world_query", "world_control", "ambiguous"]


@dataclass
class UserInput:
    raw_text: str
    source: str = "chat"  # "chat" | "auto_loop" | "system"
    timestamp: Optional[str] = None


@dataclass
class PreprocessorOutput:
    mode: Mode
    goal: str
    keywords: List[str] = field(default_factory=list)
    requires_tools: bool = False
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    memory_context: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    notes: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    analysis: str
    steps: List[str]
    tool_calls: List[ToolCall]
    response_type: Literal["text", "json"] = "text"
    needs_risk_check: bool = True
    confidence: float = 0.0


@dataclass
class CriticOutput:
    risk: Literal["low", "medium", "high", "unsafe"]
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    adjusted_steps: List[str] = field(default_factory=list)
    adjusted_tool_calls: List[ToolCall] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


@dataclass
class GovernorDecision:
    verdict: Literal["execute", "reject", "replan", "query_mode", "chat_only"]
    final_steps: List[str] = field(default_factory=list)
    final_tool_calls: List[ToolCall] = field(default_factory=list)
    final_response_mode: Literal["text", "json"] = "text"
    ask_user: Optional[str] = None
    notes: str = ""


@dataclass
class ToolResult:
    tool: str
    status: Literal["ok", "error"]
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class MemorySnapshot:
    cycle_index: int
    mode: Mode
    goal: str
    plan: List[str]
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    state: Dict[str, Any]
    notes: str = ""


@dataclass
class AgentCycleOutput:
    user: UserInput
    preprocessor: PreprocessorOutput
    plan: Plan
    critic: CriticOutput
    governor: GovernorDecision
    tool_results: List[ToolResult]
    updated_state: Dict[str, Any]


__all__ = [
    "Mode",
    "UserInput",
    "PreprocessorOutput",
    "ToolCall",
    "Plan",
    "CriticOutput",
    "GovernorDecision",
    "ToolResult",
    "MemorySnapshot",
    "AgentCycleOutput",
]
