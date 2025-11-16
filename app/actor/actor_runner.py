"""Actor runner responsible for calling the language model and parsing output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .actor_prompt import ACTOR_PROMPT

actor_prompt_template = ACTOR_PROMPT + """

OBJECTIVES:
{objectives}

CONTEXT:
{context}

MEMORY:
{memory}
"""


class SupportsGenerate(Protocol):
    """Protocol describing the language model client used by the actor."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        """Return the raw text completion for ``prompt``."""


@dataclass
class ToolCall:
    """Representation of a single tool invocation proposed by the actor."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActorOutput:
    """Structured output returned by the actor."""

    analysis: str
    plan: List[str]
    tool_calls: List[ToolCall]
    information_gaps: List[str]
    confidence: float

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "ActorOutput":
        tool_calls = [
            ToolCall(tool_name=tc.get("tool_name", ""), arguments=tc.get("arguments", {}))
            for tc in payload.get("tool_calls", [])
        ]
        return cls(
            analysis=payload.get("analysis", ""),
            plan=list(payload.get("plan", [])),
            tool_calls=tool_calls,
            information_gaps=list(payload.get("information_gaps", [])),
            confidence=float(payload.get("confidence", 0.0)),
        )


class ActorRunner:
    """Coordinates prompt building, model invocation, and response parsing."""

    def __init__(self, client: SupportsGenerate):
        self._client = client

    def run(
        self,
        objectives: List[str],
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> ActorOutput:
        ctx = context or {}
        objectives_block = json.dumps(objectives, indent=2)
        context_block = json.dumps(context or {}, indent=2)
        memory_block = json.dumps(memory_snippets or [], indent=2)
        prompt = actor_prompt_template.format(
            objectives=objectives_block,
            context=context_block,
            memory=memory_block,
        )
        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)
        output = ActorOutput.from_json(payload)
        if ctx.get("force_action") and not output.tool_calls:
            raise ValueError("Actor must produce a tool call in force_action mode.")
        return output


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse the model response and surface helpful errors."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Actor returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["ActorRunner", "ActorOutput", "ToolCall"]
