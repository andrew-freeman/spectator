"""Renderer utilities for translating agent outputs to human-readable responses."""
from __future__ import annotations

from typing import List

from app.core.schemas import AgentCycleOutput, ToolResult


def _format_tool_results(results: List[ToolResult]) -> str:
    if not results:
        return ""
    lines = []
    for result in results:
        if result.status == "ok":
            if result.result:
                lines.append(f"- {result.tool}: {result.result}")
            else:
                lines.append(f"- {result.tool}: ok")
        else:
            lines.append(f"- {result.tool}: ERROR: {result.error}")
    return "\n".join(lines)


def render_chat_response(output: AgentCycleOutput) -> str:
    mode = output.preprocessor.mode
    plan = output.plan
    gov = output.governor
    tools = output.tool_results

    if gov.ask_user:
        return gov.ask_user

    if mode == "chat":
        if not plan.steps:
            return (
                "I am Spectator, a local, reflection-driven assistant monitoring this workstation "
                "and carrying out safe, policy-aware actions. "
                f"You said: '{output.user.raw_text}'."
            )
        return plan.steps[0]

    if mode == "knowledge":
        if plan.steps:
            return plan.steps[0]
        return "Here's my best answer: " + plan.analysis

    if mode == "world_query":
        if tools:
            formatted = _format_tool_results(tools)
            return f"Here is what I found based on the requested readings:\n{formatted}"
        return "I attempted to read the system state, but no tool results were available."

    if mode == "world_control":
        lines = []
        if plan.analysis:
            lines.append("Analysis: " + plan.analysis)
        if plan.steps:
            lines.append("Planned steps: " + "; ".join(plan.steps))
        if tools:
            lines.append(_format_tool_results(tools))
        if not lines:
            lines.append("Action completed.")
        return "\n\n".join(lines)

    if plan.steps:
        return plan.steps[0]
    return plan.analysis or f"I processed your request: '{output.user.raw_text}'."


__all__ = ["render_chat_response"]
