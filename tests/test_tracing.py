import json
from pathlib import Path

from spectator.core.tracing import TraceEvent, TraceWriter


def test_trace_writer_emits_jsonl(tmp_path: Path) -> None:
    writer = TraceWriter("session-1", base_dir=tmp_path / "data" / "traces")
    event = TraceEvent(kind="note", ts=123.0, data={"ok": True})

    path = writer.write(event)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["kind"] == "note"
    assert parsed["ts"] == 123.0
    assert parsed["data"] == {"ok": True}
