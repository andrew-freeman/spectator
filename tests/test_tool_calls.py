import json

from spectator.runtime import tool_calls


def test_extract_tool_calls_parses_and_strips_block() -> None:
    payload = [
        {"name": "shell.exec", "arguments": {"cmd": "echo hi"}},
        {"name": "http.get", "arguments": {"url": "https://example.com"}},
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
    assert [call.name for call in calls] == ["shell.exec", "http.get"]
    assert calls[0].arguments == {"cmd": "echo hi"}


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
