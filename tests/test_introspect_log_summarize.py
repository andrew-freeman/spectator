from __future__ import annotations

import json
from pathlib import Path

from spectator.analysis.chunking import chunk_file
from spectator.analysis.introspection import summarize_repo_file


def test_introspect_log_summarize_uses_tail(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "traces").mkdir()

    log_rel_path = "tests/fixtures/summ.log"
    log_path = repo_root / log_rel_path
    text = log_path.read_text(encoding="utf-8")
    chunks = chunk_file(log_rel_path, text, strategy="log", max_chars=40000)
    log_chunks = [chunk for chunk in chunks if chunk.title.startswith("log ")]
    nonlog_chunks = [chunk for chunk in chunks if not chunk.title.startswith("log ")]

    responses: list[str] = []
    responses.extend(["log map"] * len(log_chunks))
    if log_chunks:
        responses.append("Initialization details: model loaded.")
    responses.extend(["nonlog map"] * len(nonlog_chunks))
    if nonlog_chunks:
        responses.append("Non-log tail: I'm doing great - thanks for asking!")

    monkeypatch.setenv(
        "SPECTATOR_FAKE_ROLE_RESPONSES",
        json.dumps({"governor": responses}),
    )

    result = summarize_repo_file(
        repo_root,
        log_rel_path,
        data_root=data_root,
        backend_name="fake",
        chunking="log",
        max_chars=40000,
    )

    summary = result["summary"]
    assert "**Log Summary**" in summary
    assert "Initialization details" in summary
    assert "**Non-log Tail**" in summary
    assert "I'm doing great - thanks for asking!" in summary
    log_section, tail_section = summary.split("**Non-log Tail**", 1)
    assert "I'm doing great - thanks for asking!" not in log_section
