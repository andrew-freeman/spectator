from __future__ import annotations

from pathlib import Path

from spectator.analysis import introspection
from spectator.analysis.chunking import chunk_file
from spectator.backends.fake import FakeBackend


def test_introspect_summarize_long_file(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "traces").mkdir()

    line = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n"
    text = line * 6000
    sample = repo_root / "long.txt"
    sample.write_text(text, encoding="utf-8")

    max_chars = 20000
    chunks = chunk_file("long.txt", text, strategy="fixed", max_chars=max_chars)
    assert len(chunks) > 1

    backend = FakeBackend()
    backend.extend_role_responses(
        "governor",
        ["chunk summary"] * len(chunks) + ["final summary"],
    )
    monkeypatch.setattr(introspection, "get_backend", lambda _name: backend)

    result = introspection.summarize_repo_file(
        repo_root,
        "long.txt",
        data_root=data_root,
        backend_name="fake",
        chunking="fixed",
        max_chars=max_chars,
    )

    assert len(backend.calls) >= len(chunks) + 1
    assert all(len(call["prompt"]) < 80000 for call in backend.calls)
    assert "Chunks:" in result["summary"]
