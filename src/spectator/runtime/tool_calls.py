from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

START_MARKER = "<<<TOOL_CALLS_JSON>>>"
END_MARKER = "<<<END_TOOL_CALLS_JSON>>>"


@dataclass(slots=True)
class ToolCall:
    id: str
    tool: str
    args: dict[str, Any]


def _extract_block(text: str) -> tuple[str | None, int, int]:
    start_index = text.find(START_MARKER)
    if start_index == -1:
        return None, -1, -1
    end_index = text.find(END_MARKER, start_index)
    if end_index == -1:
        return None, -1, -1
    payload_start = start_index + len(START_MARKER)
    payload = text[payload_start:end_index].strip()
    return payload, start_index, end_index + len(END_MARKER)


def _coerce_tool_calls(data: Any) -> list[ToolCall] | None:
    items: list[dict[str, Any]]
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            return None
        items = data
    else:
        return None

    tool_calls: list[ToolCall] = []
    for item in items:
        call_id = item.get("id")
        tool = item.get("tool")
        args = item.get("args")
        if not isinstance(call_id, str) or not isinstance(tool, str) or not isinstance(args, dict):
            return None
        tool_calls.append(ToolCall(id=call_id, tool=tool, args=args))
    return tool_calls


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    payload, start_index, end_index = _extract_block(text)
    if payload is None:
        return text, []

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return text, []

    tool_calls = _coerce_tool_calls(data)
    if tool_calls is None:
        return text, []

    visible_text = f"{text[:start_index]}{text[end_index:]}"
    return visible_text, tool_calls
