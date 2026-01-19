from __future__ import annotations

from pathlib import Path

import pytest

from spectator.tools.fs_tools import list_dir_handler, read_text_handler, write_text_handler


def test_fs_read_list_write(tmp_path: Path) -> None:
    root = tmp_path
    write_handler = write_text_handler(root)
    read_handler = read_text_handler(root)
    list_handler = list_dir_handler(root)

    result = write_handler({"path": "notes/hello.txt", "text": "hello"})
    assert result["bytes"] > 0

    read_result = read_handler({"path": "notes/hello.txt"})
    assert read_result["text"] == "hello"

    list_result = list_handler({"path": "notes"})
    assert list_result["entries"] == ["hello.txt"]


def test_fs_write_overwrite_rules(tmp_path: Path) -> None:
    root = tmp_path
    write_handler = write_text_handler(root)

    write_handler({"path": "data.txt", "text": "one"})
    with pytest.raises(ValueError):
        write_handler({"path": "data.txt", "text": "two"})

    overwrite_result = write_handler({
        "path": "data.txt",
        "text": "two",
        "overwrite": True,
    })
    assert overwrite_result["bytes"] > 0


def test_fs_escape_denied(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write_handler = write_text_handler(root)

    with pytest.raises(ValueError):
        write_handler({"path": "../escape.txt", "text": "nope"})


def test_fs_accepts_sandbox_alias(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "hello.txt").write_text("hello", encoding="utf-8")
    list_handler = list_dir_handler(root)

    result = list_handler({"path": "/sandbox"})
    assert "hello.txt" in result["entries"]
