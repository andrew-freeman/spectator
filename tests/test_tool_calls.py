import json

from spectator.runtime import tool_calls


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
