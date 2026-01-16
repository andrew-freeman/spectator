from __future__ import annotations

from pathlib import Path

import pytest

from spectator.tools.shell_tool import shell_exec_handler


def test_shell_allows_echo(tmp_path: Path) -> None:
    handler = shell_exec_handler(tmp_path)
    result = handler({"cmd": "echo hi"})
    assert "hi" in result["stdout"]


def test_shell_denies_rm(tmp_path: Path) -> None:
    handler = shell_exec_handler(tmp_path)
    with pytest.raises(ValueError):
        handler({"cmd": "rm -rf /"})


def test_shell_timeout(tmp_path: Path) -> None:
    handler = shell_exec_handler(tmp_path)
    with pytest.raises(RuntimeError):
        handler({
            "cmd": "python -c \"import time; time.sleep(2)\"",
            "timeout_s": 0.1,
        })
