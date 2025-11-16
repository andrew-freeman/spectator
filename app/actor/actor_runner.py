"""Actor runner responsible for calling the language model and parsing output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .actor_prompt import ACTOR_PROMPT

actor_prompt_template = ACTOR_PROMPT + """

IDENTITY:
{identity}

POLICY_GUIDANCE:
{policy}

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

    def __init__(self, client: SupportsGenerate, *, identity: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity or {}
        self._policy = policy or {}

    def run(
        self,
        objectives: List[str],
        context: Optional[Dict[str, Any]] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> ActorOutput:
        ctx = context or {}
        query_type = ctx.get("query_type")
        objectives_block = json.dumps(objectives, indent=2)
        context_block = json.dumps(ctx, indent=2)
        memory_block = json.dumps(memory_snippets or [], indent=2)
        identity_block = json.dumps(self._identity, indent=2)
        policy_block = json.dumps(self._policy, indent=2)
        prompt = actor_prompt_template.format(
            objectives=objectives_block,
            context=context_block,
            memory=memory_block,
            identity=identity_block,
            policy=policy_block,
        )
        raw = self._client.generate(prompt, stop=None)

        payload = _parse_json(raw)
        output = ActorOutput.from_json(payload)
        if query_type == "knowledge":
            if output.tool_calls:
                output.tool_calls = []
            return output
        if ctx.get("force_action") and not output.tool_calls:
            logging.getLogger(__name__).warning(
                "force_action requested but actor produced no tool calls; falling back to reasoning."
            )
        return output


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse the model response and surface helpful errors."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Actor returned invalid JSON: {exc}: {raw!r}") from exc


__all__ = ["ActorRunner", "ActorOutput", "ToolCall"]
