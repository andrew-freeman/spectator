from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .responder_prompt import RESPONDER_PROMPT


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class ResponderOutput:
    final_text: str
    short_summary: str
    used_tools: List[str]
    mode: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ResponderOutput":
        final_text = str(payload.get("final_text", "")).strip()
        short_summary = str(payload.get("short_summary", "")).strip()
        used_tools_raw = payload.get("used_tools", [])
        if not isinstance(used_tools_raw, list):
            used_tools = []
        else:
            used_tools = [str(x) for x in used_tools_raw]
        mode = str(payload.get("mode", "")).strip() or "chat"
        return cls(
            final_text=final_text,
            short_summary=short_summary,
            used_tools=used_tools,
            mode=mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResponderRunner:
    """Generate the final user-facing response based on plan + tools + state."""

    def __init__(self, client: SupportsGenerate, identity: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity or {}

    def run(
        self,
        *,
        mode: str,
        user_message: str,
        reflection: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
        governor: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reflection = reflection or {}
        plan = plan or {}
        governor = governor or {}
        tool_results = tool_results or []
        state = state or {}

        input_payload = {
            "mode": mode,
            "user_message": user_message,
            "reflection": reflection,
            "plan": plan,
            "governor": governor,
            "tool_results": tool_results,
            "state": state,
            "identity": self._identity,
        }

        # Build prompt WITHOUT format() to avoid brace issues.
        prompt = (
            RESPONDER_PROMPT
            + "\n\nINPUT JSON:\n"
            + json.dumps(input_payload, indent=2)
        )

        try:
            raw = self._client.generate(prompt, stop=None)
            payload = json.loads(raw)
            output = ResponderOutput.from_payload(payload)
        except Exception:
            # Very defensive fallback.
            output = ResponderOutput(
                final_text="I encountered an internal issue while forming the reply, but the underlying tools and reasoning still executed.",
                short_summary="Fallback responder output.",
                used_tools=[],
                mode=mode or "chat",
            )

        return output.to_dict()


__all__ = ["ResponderRunner", "ResponderOutput"]
