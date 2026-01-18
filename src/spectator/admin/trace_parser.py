from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_PROMPT_HEADER_RE = re.compile(r"^([A-Z_]+):\s*$")


def _parse_prompt_sections(prompt: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_header: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if current_header is None:
            return
        text = "\n".join(buffer).strip("\n")
        sections[current_header] = text
        buffer = []

    for line in prompt.splitlines():
        match = _PROMPT_HEADER_RE.match(line)
        if match:
            flush()
            current_header = match.group(1)
            continue
        if current_header is not None:
            buffer.append(line)
    flush()
    return sections


def _pretty_history_json(raw_text: str) -> str:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _parse_line(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_role(event: dict[str, Any]) -> str | None:
    data = event.get("data")
    if isinstance(data, dict):
        role = data.get("role")
        if isinstance(role, str):
            return role
    return None


def parse_trace_file(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    events_by_role: dict[str, list[dict[str, Any]]] = {}
    per_role: dict[str, dict[str, Any]] = {}
    tool_calls: dict[str, dict[str, Any]] = {}
    sanitize_events: list[dict[str, Any]] = []
    sanitize_warnings: list[dict[str, Any]] = []
    final_visible_response: str | None = None

    if not path.exists():
        return {
            "events": [],
            "events_by_role": {},
            "per_role": [],
            "tool_calls": [],
            "sanitize": [],
            "sanitize_warnings": [],
            "final_response": None,
        }

    for line in path.read_text(encoding="utf-8").splitlines():
        payload = _parse_line(line)
        if payload is None:
            continue
        ts = payload.get("ts")
        kind = payload.get("kind")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        role = _extract_role(payload)

        event = {
            "ts": ts,
            "kind": kind,
            "role": role,
            "data": data,
        }
        events.append(event)
        if role is not None:
            events_by_role.setdefault(role, []).append(event)

        if kind in {"llm_req", "llm_done"} and role is not None:
            entry = per_role.setdefault(
                role,
                {
                    "role": role,
                    "llm_req": None,
                    "llm_done": None,
                    "prompt_sections": {},
                    "history_json_pretty": None,
                },
            )
            if kind == "llm_req":
                entry["llm_req"] = event
                prompt = data.get("prompt")
                if isinstance(prompt, str):
                    sections = _parse_prompt_sections(prompt)
                    entry["prompt_sections"] = sections
                    history_section = sections.get("HISTORY_JSON")
                    if isinstance(history_section, str):
                        entry["history_json_pretty"] = _pretty_history_json(
                            history_section
                        )
            if kind == "llm_done":
                entry["llm_done"] = event

        if kind == "tool_start" and isinstance(data, dict):
            tool_id = data.get("id")
            if isinstance(tool_id, str):
                tool_calls[tool_id] = {
                    "id": tool_id,
                    "tool": data.get("tool"),
                    "role": role,
                    "args": data.get("args"),
                    "ok": None,
                    "error": None,
                    "duration_ms": None,
                    "metadata": {},
                }
        if kind == "tool_done" and isinstance(data, dict):
            tool_id = data.get("id")
            if isinstance(tool_id, str):
                entry = tool_calls.setdefault(
                    tool_id,
                    {
                        "id": tool_id,
                        "tool": data.get("tool"),
                        "role": role,
                        "args": data.get("args"),
                        "ok": None,
                        "error": None,
                        "duration_ms": None,
                        "metadata": {},
                    },
                )
                entry.update(
                    {
                        "tool": data.get("tool", entry.get("tool")),
                        "role": role or entry.get("role"),
                        "args": data.get("args", entry.get("args")),
                        "ok": data.get("ok"),
                        "error": data.get("error"),
                        "duration_ms": data.get("duration_ms"),
                    }
                )
                metadata = {
                    key: value
                    for key, value in data.items()
                    if key
                    not in {
                        "id",
                        "tool",
                        "role",
                        "args",
                        "ok",
                        "error",
                        "duration_ms",
                    }
                }
                if metadata:
                    entry["metadata"] = metadata

        if kind == "sanitize" and isinstance(data, dict):
            sanitize_events.append(data)
        if kind == "sanitize_warning" and isinstance(data, dict):
            sanitize_warnings.append(data)
        if kind == "visible_response" and isinstance(data, dict):
            visible = data.get("visible_response")
            if isinstance(visible, str):
                final_visible_response = visible

    tool_calls_list = list(tool_calls.values())
    tool_calls_list.sort(key=lambda item: item.get("id") or "")
    per_role_list = list(per_role.values())
    per_role_list.sort(key=lambda item: item.get("role") or "")

    return {
        "events": events,
        "events_by_role": events_by_role,
        "per_role": per_role_list,
        "tool_calls": tool_calls_list,
        "sanitize": sanitize_events,
        "sanitize_warnings": sanitize_warnings,
        "final_response": final_visible_response,
    }
