from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectator.admin.app import create_app

pytestmark = pytest.mark.admin


def _write_checkpoint(path: Path, session_id: str, trace_tail: list[str]) -> None:
    payload = {
        "session_id": session_id,
        "revision": 1,
        "updated_ts": 123.0,
        "state": {},
        "recent_messages": [],
        "trace_tail": trace_tail,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trace(path: Path) -> None:
    events = [
        {
            "ts": 1.0,
            "kind": "tool_start",
            "data": {"role": "governor", "id": "tool-1", "tool": "http_get", "args": {"url": "https://example.com"}},
        },
        {
            "ts": 2.0,
            "kind": "tool_done",
            "data": {"role": "governor", "id": "tool-1", "tool": "http_get", "ok": True, "error": None, "duration_ms": 8.0},
        },
        {
            "ts": 2.5,
            "kind": "llm_req",
            "data": {
                "role": "planner",
                "system_prompt": "system prompt",
                "prompt": "STATE:\nstate payload\nUSER:\nuser payload",
            },
        },
        {
            "ts": 2.8,
            "kind": "llm_done",
            "data": {"role": "planner", "response": "response payload"},
        },
        {
            "ts": 3.0,
            "kind": "visible_response",
            "data": {"role": "governor", "visible_response": "Done"},
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_admin_api_lists_sessions_and_runs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    checkpoint_dir = data_root / "checkpoints"
    trace_dir = data_root / "traces"
    checkpoint_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    session_id = "session-1"
    run_id = "rev-1"
    trace_name = f"{session_id}__{run_id}.jsonl"

    _write_checkpoint(checkpoint_dir / f"{session_id}.json", session_id, [trace_name])
    _write_trace(trace_dir / trace_name)

    client = TestClient(create_app(data_root=data_root))

    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["session_id"] == session_id

    runs = client.get(f"/api/sessions/{session_id}/runs").json()["runs"]
    assert runs[0]["run_id"] == run_id

    run_detail = client.get(f"/api/sessions/{session_id}/runs/{run_id}").json()
    assert run_detail["final_response"] == "Done"
    assert run_detail["tool_calls"][0]["tool"] == "http_get"
    assert run_detail["per_role"][0]["prompt_sections"]["USER"] == "user payload"
    assert run_detail["per_role"][0]["llm_done"]["data"]["response"] == "response payload"
