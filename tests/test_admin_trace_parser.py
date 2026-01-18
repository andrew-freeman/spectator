from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from spectator.admin.trace_parser import parse_trace_file

pytestmark = pytest.mark.admin


def test_parse_trace_file_extracts_summary(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {
            "ts": 1.0,
            "kind": "tool_start",
            "data": {"role": "governor", "id": "tool-1", "tool": "http_get", "args": {"url": "https://example.com"}},
        },
        {
            "ts": 2.0,
            "kind": "tool_done",
            "data": {"role": "governor", "id": "tool-1", "tool": "http_get", "ok": True, "error": None, "duration_ms": 12.5},
        },
        {
            "ts": 3.0,
            "kind": "sanitize_warning",
            "data": {"role": "governor", "message": "visible output empty after sanitization"},
        },
        {
            "ts": 4.0,
            "kind": "visible_response",
            "data": {"role": "governor", "visible_response": "Hello"},
        },
    ]
    trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    parsed = parse_trace_file(trace_path)

    assert parsed["final_response"] == "Hello"
    assert parsed["sanitize_warnings"][0]["message"] == "visible output empty after sanitization"
    assert parsed["tool_calls"][0]["id"] == "tool-1"
    assert parsed["tool_calls"][0]["ok"] is True


def test_parse_trace_file_parses_prompt_sections(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    prompt = "\n".join(
        [
            "STATE:",
            "state payload",
            "HISTORY_JSON:",
            '[{"role": "user", "content": "hi"}]',
            "UPSTREAM:",
            "upstream payload",
            "USER:",
            "user payload",
        ]
    )
    events = [
        {
            "ts": 1.0,
            "kind": "llm_req",
            "data": {
                "role": "planner",
                "prompt": prompt,
                "system_prompt": "system prompt",
            },
        },
        {
            "ts": 2.0,
            "kind": "llm_done",
            "data": {"role": "planner", "response": "ok"},
        },
    ]
    trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    parsed = parse_trace_file(trace_path)

    per_role = parsed["per_role"][0]
    assert per_role["role"] == "planner"
    assert set(per_role["prompt_sections"].keys()) == {"STATE", "HISTORY_JSON", "UPSTREAM", "USER"}
    assert per_role["history_json_pretty"] == json.dumps(
        [{"role": "user", "content": "hi"}],
        indent=2,
        ensure_ascii=False,
    )


def test_parse_trace_file_preserves_invalid_history_json(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    prompt = "\n".join(
        [
            "STATE:",
            "state payload",
            "HISTORY_JSON:",
            "{invalid json]",
            "USER:",
            "user payload",
        ]
    )
    events = [
        {
            "ts": 1.0,
            "kind": "llm_req",
            "data": {
                "role": "reflection",
                "prompt": prompt,
                "system_prompt": "system prompt",
            },
        }
    ]
    trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    parsed = parse_trace_file(trace_path)

    per_role = parsed["per_role"][0]
    assert per_role["prompt_sections"]["HISTORY_JSON"] == "{invalid json]"
    assert per_role["history_json_pretty"] == "{invalid json]"
