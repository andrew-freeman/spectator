from spectator.runtime.notes import END_MARKER as NOTES_END
from spectator.runtime.notes import START_MARKER as NOTES_START
from spectator.runtime.sanitize import sanitize_visible_text
from spectator.runtime.tool_calls import END_MARKER as TOOLS_END
from spectator.runtime.tool_calls import START_MARKER as TOOLS_START


def test_sanitize_strips_think_blocks() -> None:
    text = "Hello <think>hidden</think> world"
    assert sanitize_visible_text(text) == "Hello  world"


def test_sanitize_strips_notes_block() -> None:
    text = (
        "Intro\n"
        f"{NOTES_START}\n"
        '{"note": "<think>keep</think>"}\n'
        f"{NOTES_END}\n"
        "Outro"
    )
    assert sanitize_visible_text(text) == "Intro\n\nOutro"


def test_sanitize_strips_tool_calls_block() -> None:
    text = (
        "Intro\n"
        f"{TOOLS_START}\n"
        '[{"id": "t1", "tool": "fs.list_dir", "args": {"path": "<think>./</think>"}}]\n'
        f"{TOOLS_END}\n"
        "Outro"
    )
    assert sanitize_visible_text(text) == "Intro\n\nOutro"


def test_sanitize_strips_bare_tool_call_json() -> None:
    text = '{"name":"fs.list_dir","arguments":"{\\"path\\":\\"/sandbox\\"}"}'
    assert sanitize_visible_text(text) == "..."


def test_sanitize_strips_state_only_output() -> None:
    text = "STATE:\n{...}"
    assert sanitize_visible_text(text) == "..."


def test_sanitize_strips_history_only_output() -> None:
    text = "HISTORY_JSON:\n[{\"role\": \"user\", \"content\": \"hi\"}]"
    assert sanitize_visible_text(text) == "..."


def test_sanitize_strips_trailing_state_block() -> None:
    text = "Hello\n\nSTATE:\n{...}"
    assert sanitize_visible_text(text) == "Hello"


def test_sanitize_strips_trailing_history_block() -> None:
    text = "Hello\n\nHISTORY_JSON:\n[{\"role\": \"user\", \"content\": \"hi\"}]"
    assert sanitize_visible_text(text) == "Hello"


def test_sanitize_strips_leading_role_transcript_block() -> None:
    text = "reflection:\nThoughts\n\nHello"
    assert sanitize_visible_text(text) == "Hello"


def test_sanitize_strips_trailing_role_transcript_block() -> None:
    text = "Hello\n\nassistant:\nOops"
    assert sanitize_visible_text(text) == "Hello"


def test_sanitize_keeps_notes_json_marker() -> None:
    text = f"{NOTES_START}\n{{\"notes\": true}}\n{NOTES_END}"
    assert sanitize_visible_text(text) == "..."


def test_sanitize_keeps_tool_calls_json_marker() -> None:
    text = f"{TOOLS_START}\n[]\n{TOOLS_END}"
    assert sanitize_visible_text(text) == "..."


def test_sanitize_strips_dangling_markers() -> None:
    text = f"Alpha {TOOLS_START} Beta {NOTES_END} Gamma"
    assert sanitize_visible_text(text) == "Alpha  Beta  Gamma"


def test_sanitize_strips_dangling_markers_only_output() -> None:
    text = f"{TOOLS_START}\n{NOTES_END}"
    assert sanitize_visible_text(text) == "..."


def test_sanitize_strips_retrieval_block() -> None:
    text = (
        "Hello\n"
        "=== RETRIEVAL ===\n"
        "[1] id=mem-1 text=remember\n"
        "=== END RETRIEVAL ===\n"
        "World"
    )
    assert sanitize_visible_text(text) == "Hello\n\nWorld"


def test_sanitize_strips_retrieved_memory_block() -> None:
    text = (
        "Alpha\n"
        "=== RETRIEVED_MEMORY ===\n"
        "something\n"
        "=== END_RETRIEVED_MEMORY ===\n"
        "Omega"
    )
    assert sanitize_visible_text(text) == "Alpha\n\nOmega"
