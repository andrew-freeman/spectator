from __future__ import annotations

import json
from pathlib import Path

from spectator.analysis.introspection import (
    list_repo_files,
    read_repo_file_tail,
    summarize_repo_file,
)


def test_introspection_list_and_read(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sample = repo_root / "sample.txt"
    sample.write_text("line1\nline2\nline3\n", encoding="utf-8")

    files = list_repo_files(repo_root, limit=10)
    assert "sample.txt" in files

    tail = read_repo_file_tail(repo_root, "sample.txt", max_lines=2)
    assert tail == "line2\nline3"


def test_introspection_summarize_uses_fake_backend(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "traces").mkdir()

    sample = repo_root / "sample.txt"
    sample.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    responses = {"governor": ["Chunk summary.", "Summary here."]}
    monkeypatch.setenv("SPECTATOR_FAKE_ROLE_RESPONSES", json.dumps(responses))

    result = summarize_repo_file(
        repo_root,
        "sample.txt",
        data_root=data_root,
        backend_name="fake",
        max_lines=2,
        instruction="Summarize.",
    )

    assert "**Log Summary**" in result["summary"]
    assert "**Non-log Tail**" in result["summary"]
    assert "Summary here." in result["summary"]
    assert "Chunks:" in result["summary"]
