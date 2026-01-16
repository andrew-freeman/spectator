from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class State:
    goals: List[str] = field(default_factory=list)
    open_loops: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    episode_summary: str = ""
    memory_tags: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)
    capabilities_granted: List[str] = field(default_factory=list)
    capabilities_pending: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class Checkpoint:
    session_id: str
    revision: int
    updated_ts: float
    state: State
    recent_messages: List[ChatMessage] = field(default_factory=list)
    trace_tail: List[str] = field(default_factory=list)
