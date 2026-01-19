from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from spectator.core.tracing import TraceEvent, TraceWriter

START_MARKER = "<<<TOOL_CALLS_JSON>>>"
END_MARKER = "<<<END_TOOL_CALLS_JSON>>>"
DEFAULT_ALLOWED_PREFIXES = ("fs.", "shell.", "http.")


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


def _is_allowed_tool(
    name: str,
    allowed_tools: set[str] | None,
    allowed_prefixes: tuple[str, ...],
) -> bool:
    if allowed_tools and name in allowed_tools:
        return True
    return any(name.startswith(prefix) for prefix in allowed_prefixes)


def _emit_trace(
    tracer: TraceWriter | None,
    role: str | None,
    kind: str,
    data: dict[str, Any],
) -> None:
    if tracer is None:
        return
    payload = dict(data)
    if role is not None:
        payload["role"] = role
    tracer.write(
        TraceEvent(
            ts=time.time(),
            kind=kind,
            data=payload,
        )
    )


def _parse_args(value: Any, warnings: list[str]) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            warnings.append("arguments_json_invalid")
            return None
        if not isinstance(parsed, dict):
            warnings.append("arguments_not_object")
            return None
        return parsed
    warnings.append("arguments_type_invalid")
    return None


def _coerce_loose_tool_calls(
    data: Any,
    *,
    allowed_tools: set[str] | None,
    allowed_prefixes: tuple[str, ...],
    tracer: TraceWriter | None,
    role: str | None,
) -> list[ToolCall]:
    warnings: list[str] = []
    items: list[dict[str, Any]]
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            warnings.append("payload_items_not_objects")
            _emit_parse_warnings(tracer, role, warnings)
            return []
        items = data
    else:
        warnings.append("payload_not_object_or_list")
        _emit_parse_warnings(tracer, role, warnings)
        return []

    tool_calls: list[ToolCall] = []
    formats: set[str] = set()
    auto_index = 1
    for item in items:
        tool_key = None
        tool_value = item.get("tool")
        if isinstance(tool_value, str):
            tool_key = "tool"
        else:
            tool_value = item.get("name")
            if isinstance(tool_value, str):
                tool_key = "name"
        if tool_key is None:
            warnings.append("missing_tool")
            continue

        args_key = None
        if "args" in item:
            args_key = "args"
            raw_args = item.get("args")
        elif "arguments" in item:
            args_key = "arguments"
            raw_args = item.get("arguments")
        else:
            warnings.append("missing_arguments")
            continue

        formats.add(f"{tool_key}/{args_key}")
        if not _is_allowed_tool(tool_value, allowed_tools, allowed_prefixes):
            warnings.append("tool_not_allowed")
            continue

        args = _parse_args(raw_args, warnings)
        if args is None:
            continue

        call_id = item.get("id") if isinstance(item.get("id"), str) else None
        if call_id is None:
            call_id = f"auto-{auto_index}"
            auto_index += 1
        tool_calls.append(ToolCall(id=call_id, tool=tool_value, args=args))

    if tool_calls:
        if len(formats) == 1:
            original_format = next(iter(formats))
        else:
            original_format = "mixed"
        _emit_trace(
            tracer,
            role,
            "tool_calls_coerced",
            {"original_format": original_format, "count": len(tool_calls)},
        )
    _emit_parse_warnings(tracer, role, warnings)
    return tool_calls


def _emit_parse_warnings(
    tracer: TraceWriter | None, role: str | None, warnings: list[str]
) -> None:
    for warning in warnings:
        _emit_trace(tracer, role, "tool_calls_parse_warning", {"reason": warning})


def extract_tool_calls(
    text: str,
    *,
    tracer: TraceWriter | None = None,
    role: str | None = None,
    allowed_tools: set[str] | None = None,
    allowed_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_PREFIXES,
) -> tuple[str, list[ToolCall]]:
    payload, start_index, end_index = _extract_block(text)
    if payload is None:
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return text, []
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            _emit_trace(
                tracer,
                role,
                "tool_calls_parse_warning",
                {"reason": "payload_json_invalid"},
            )
            return text, []
        tool_calls = _coerce_loose_tool_calls(
            data,
            allowed_tools=allowed_tools,
            allowed_prefixes=allowed_prefixes,
            tracer=tracer,
            role=role,
        )
        if not tool_calls:
            return text, []
        return "", tool_calls

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return text, []

    tool_calls = _coerce_tool_calls(data)
    if tool_calls is None:
        return text, []

    filtered_calls: list[ToolCall] = []
    for call in tool_calls:
        if _is_allowed_tool(call.tool, allowed_tools, allowed_prefixes):
            filtered_calls.append(call)
        else:
            _emit_trace(
                tracer,
                role,
                "tool_calls_parse_warning",
                {"reason": "tool_not_allowed"},
            )

    visible_text = f"{text[:start_index]}{text[end_index:]}"
    return visible_text, filtered_calls
