"""Actor runner responsible for calling the language model and parsing output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .actor_prompt import ACTOR_PROMPT


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
        if ctx.get("mode") == "query":
            analysis_parts = [
                "Query mode: summarising system state without executing tools.",
            ]
            if objectives:
                analysis_parts.append(
                    "Objectives: " + "; ".join(obj.strip() for obj in objectives if obj.strip())
                )
            extra_context = {k: v for k, v in ctx.items() if k != "mode"}
            if extra_context:
                context_fragments = ", ".join(f"{k}={v}" for k, v in extra_context.items())
                analysis_parts.append(f"Context: {context_fragments}.")
            if memory_snippets:
                analysis_parts.append(
                    "Memory snippets: " + "; ".join(snippet.strip() for snippet in memory_snippets if snippet)
                )
            analysis = " ".join(analysis_parts)
            return ActorOutput(
                analysis=analysis,
                plan=[],
                tool_calls=[],
                information_gaps=[],
                confidence=1.0,
            )

        prompt = ACTOR_PROMPT
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
