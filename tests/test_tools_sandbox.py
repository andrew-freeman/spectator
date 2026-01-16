from __future__ import annotations

from pathlib import Path

import pytest

from spectator.tools.sandbox import resolve_under_root, validate_shell_cmd


def test_resolve_under_root_denies_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    assert resolve_under_root(root, "../outside") is None

    escape_symlink = root / "link"
    escape_symlink.symlink_to(outside)
    assert resolve_under_root(root, "link/target.txt") is None


def test_validate_shell_cmd_rules() -> None:
    allow = ["echo", "ls"]
    deny = ["rm", "sudo"]

    ok, reason = validate_shell_cmd("echo hi", allow, deny)
    assert ok
    assert reason is None

    ok, reason = validate_shell_cmd("rm -rf /", allow, deny)
    assert not ok
    assert reason is not None

    ok, reason = validate_shell_cmd("bash -c 'echo hi'", allow, deny)
    assert not ok
    assert reason is not None
