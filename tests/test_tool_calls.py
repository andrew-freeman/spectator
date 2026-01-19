import json
from pathlib import Path

from spectator.runtime import tool_calls
from spectator.core.tracing import TraceWriter


def test_extract_tool_calls_parses_and_strips_block() -> None:
    payload = [
        {"id": "call-1", "tool": "shell.exec", "args": {"cmd": "echo hi"}},
        {"id": "call-2", "tool": "http.get", "args": {"url": "https://example.com"}},
    ]
    text = (
        "Intro\n"
        f"{tool_calls.START_MARKER}\n"
        f"{json.dumps(payload)}\n"
        f"{tool_calls.END_MARKER}\n"
        "Outro"
    )

    visible, calls = tool_calls.extract_tool_calls(text)

    assert "TOOL_CALLS" not in visible
    assert visible.strip().startswith("Intro")
    assert visible.strip().endswith("Outro")
    assert [call.tool for call in calls] == ["shell.exec", "http.get"]
    assert calls[0].args == {"cmd": "echo hi"}


def test_extract_tool_calls_rejects_malformed_json() -> None:
    text = (
        "Before\n"
        f"{tool_calls.START_MARKER}\n"
        "{not-json]\n"
        f"{tool_calls.END_MARKER}\n"
        "After"
    )

    visible, calls = tool_calls.extract_tool_calls(text)

    assert visible == text
    assert calls == []


def test_extract_tool_calls_rejects_partial_block() -> None:
    text = (
        "Before\n"
        f"{tool_calls.START_MARKER}\n"
        "[]\n"
        "After"
    )

    visible, calls = tool_calls.extract_tool_calls(text)

    assert visible == text
    assert calls == []


def test_extract_tool_calls_accepts_bare_name_arguments_string() -> None:
    text = "{\"name\":\"fs.list_dir\",\"arguments\":\"{\\\"path\\\":\\\"/sandbox\\\"}\"}"

    visible, calls = tool_calls.extract_tool_calls(text)

    assert visible.strip() == ""
    assert len(calls) == 1
    assert calls[0].tool == "fs.list_dir"
    assert calls[0].args == {"path": "/sandbox"}


def test_extract_tool_calls_accepts_bare_arguments_dict() -> None:
    payload = {"name": "fs.list_dir", "arguments": {"path": "/sandbox"}}
    text = json.dumps(payload)

    visible, calls = tool_calls.extract_tool_calls(text)

    assert visible.strip() == ""
    assert len(calls) == 1
    assert calls[0].args == {"path": "/sandbox"}


def test_extract_tool_calls_emits_coerced_trace(tmp_path: Path) -> None:
    text = "{\"tool\":\"fs.list_dir\",\"args\":{\"path\":\"/sandbox\"}}"
    tracer = TraceWriter("session-2", base_dir=tmp_path)

    visible, calls = tool_calls.extract_tool_calls(
        text,
        tracer=tracer,
        role="governor",
    )

    assert visible.strip() == ""
    assert len(calls) == 1
    trace_lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    kinds = {json.loads(line)["kind"] for line in trace_lines}
    assert "tool_calls_coerced" in kinds


def test_extract_tool_calls_warns_on_invalid_arguments_json(tmp_path: Path) -> None:
    payload = {"name": "fs.list_dir", "arguments": "{not-json]"}
    text = json.dumps(payload)
    tracer = TraceWriter("session-1", base_dir=tmp_path)

    visible, calls = tool_calls.extract_tool_calls(
        text,
        tracer=tracer,
        role="governor",
    )

    assert visible == text
    assert calls == []
    trace_lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    kinds = {json.loads(line)["kind"] for line in trace_lines}
    assert "tool_calls_parse_warning" in kinds
