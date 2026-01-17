from __future__ import annotations

from pathlib import Path

from spectator.core.types import State
from spectator.runtime.tool_calls import ToolCall
from spectator.tools import build_default_registry


def test_executor_unknown_tool_returns_error(tmp_path: Path) -> None:
    _registry, executor = build_default_registry(tmp_path)
    state = State()
    call = ToolCall(id="t1", tool="fs.delete_tree", args={"path": "."})

    result = executor.execute_calls([call], state)[0]

    assert result.ok is False
    assert result.error == "unknown tool"


def test_executor_reports_argument_validation_errors(tmp_path: Path) -> None:
    _registry, executor = build_default_registry(tmp_path)
    state = State()
    call = ToolCall(id="t1", tool="fs.write_text", args={"path": 123, "text": "hi"})

    result = executor.execute_calls([call], state)[0]

    assert result.ok is False
    assert "path must be a string" in (result.error or "")


def test_executor_rejects_disallowed_shell_command(tmp_path: Path) -> None:
    _registry, executor = build_default_registry(tmp_path)
    state = State()
    call = ToolCall(id="t1", tool="shell.exec", args={"cmd": "rm -rf /"})

    result = executor.execute_calls([call], state)[0]

    assert result.ok is False
    assert result.error
