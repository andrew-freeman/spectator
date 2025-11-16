"""Utility helpers for rendering chat responses outside the web UI."""
from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from app.core.schemas import ToolResult

ToolPayload = Union[ToolResult, Dict[str, Any]]


def _format_tool_results(results: Sequence[ToolPayload]) -> str:
    if not results:
        return ""
    lines = []
    for result in results:
        if isinstance(result, ToolResult):
            payload = result.to_dict()
        else:
            payload = result
        tool = payload.get("tool", "tool")
        status = payload.get("status", "ok")
        if status == "ok":
            lines.append(f"- {tool}: {payload.get('result', {})}")
        else:
            lines.append(f"- {tool}: ERROR {payload.get('error', 'unknown error')}")
    return "\n".join(lines)


def render_chat_response(record: Dict[str, Any]) -> str:
    """Best-effort renderer compatible with the v2 pipeline records."""

    final_text = record.get("final_text")
    if final_text:
        return final_text

    reflection = record.get("reflection", {})
    plan = record.get("plan", {})
    governor = record.get("governor", {})
    tool_results = record.get("tool_results", [])

    mode = record.get("mode") or reflection.get("mode") or "chat"
    goal = reflection.get("goal") or record.get("user_message", "")

    if governor.get("verdict") == "reject":
        return governor.get("rationale") or "I couldn't safely perform that request."

    if mode == "world_query":
        summary = _format_tool_results(tool_results)
        if summary:
            return f"Here are the requested readings:\n{summary}"
        return "I attempted to read the system state, but no tool data was returned."

    if mode == "world_control":
        summary = _format_tool_results(tool_results)
        if summary:
            return summary
        return "I attempted the control action within policy limits."

    if plan.get("analysis"):
        return plan["analysis"]

    steps = plan.get("steps") or []
    if steps:
        return steps[0]

    return goal or "I'm ready for your next instruction."


__all__ = ["render_chat_response"]
