from __future__ import annotations

import json
from pathlib import Path

from spectator.analysis.soak import analyze_soak


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(path: Path, state_overrides: dict[str, object] | None = None) -> None:
    state = {
        "goals": [],
        "open_loops": [],
        "decisions": [],
        "constraints": [],
        "episode_summary": "",
        "memory_tags": [],
        "memory_refs": [],
        "capabilities_granted": [],
        "capabilities_pending": [],
    }
    if state_overrides:
        state.update(state_overrides)
    payload = {
        "session_id": "soak-test",
        "revision": 1,
        "updated_ts": 0.0,
        "state": state,
        "recent_messages": [],
        "trace_tail": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_analyze_soak_pass(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    events = [
        {"ts": 0.0, "kind": "llm_req", "data": {"role": "governor"}},
        {"ts": 0.1, "kind": "llm_done", "data": {"role": "governor"}},
        {
            "ts": 0.2,
            "kind": "tool_plan",
            "data": {"role": "governor", "calls": [{"id": "tool-1", "tool": "fs.read_text"}]},
        },
        {"ts": 0.3, "kind": "tool_start", "data": {"role": "governor", "id": "tool-1", "tool": "fs.read_text"}},
        {"ts": 0.4, "kind": "tool_done", "data": {"role": "governor", "id": "tool-1", "tool": "fs.read_text", "ok": True}},
        {"ts": 0.5, "kind": "notes_patch", "data": {"role": "governor"}},
    ]
    _write_trace(trace_path, events)
    _write_checkpoint(checkpoint_path)

    summary = analyze_soak(trace_path, checkpoint_path, turns=1)

    assert summary.failures == []


def test_analyze_soak_checkpoint_fail(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    events = [
        {"ts": 0.0, "kind": "llm_req", "data": {"role": "governor"}},
        {"ts": 0.1, "kind": "llm_done", "data": {"role": "governor"}},
        {
            "ts": 0.2,
            "kind": "tool_plan",
            "data": {"role": "governor", "calls": [{"id": "tool-1", "tool": "fs.read_text"}]},
        },
        {"ts": 0.3, "kind": "tool_start", "data": {"role": "governor", "id": "tool-1", "tool": "fs.read_text"}},
        {"ts": 0.4, "kind": "tool_done", "data": {"role": "governor", "id": "tool-1", "tool": "fs.read_text", "ok": True}},
        {"ts": 0.5, "kind": "notes_patch", "data": {"role": "governor"}},
    ]
    _write_trace(trace_path, events)
    _write_checkpoint(
        checkpoint_path,
        state_overrides={"capabilities_granted": ["net"], "capabilities_pending": ["net"]},
    )

    summary = analyze_soak(trace_path, checkpoint_path, turns=1)

    assert any("Capabilities pending intersect with granted" in item for item in summary.failures)
