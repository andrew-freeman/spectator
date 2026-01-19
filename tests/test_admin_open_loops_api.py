from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectator.admin.app import create_app
from spectator.runtime import checkpoints

pytestmark = pytest.mark.admin


def _write_checkpoint(path: Path, session_id: str) -> None:
    payload = {
        "session_id": session_id,
        "revision": 1,
        "updated_ts": 123.0,
        "state": {"open_loops": []},
        "recent_messages": [],
        "trace_tail": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_admin_open_loops_crud(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    checkpoint_dir = data_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    session_id = "session-1"
    _write_checkpoint(checkpoint_dir / f"{session_id}.json", session_id)

    client = TestClient(create_app(data_root=data_root))

    create_resp = client.post(
        f"/api/sessions/{session_id}/open_loops",
        json={
            "title": "Fix sandbox path",
            "details": "Alias /sandbox to root",
            "tags": ["fs", "admin"],
            "priority": 2,
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()["open_loops"]
    assert len(created) == 1
    loop_id = created[0]["id"]
    assert loop_id == "loop-1"

    list_resp = client.get(f"/api/sessions/{session_id}/open_loops")
    assert list_resp.status_code == 200
    listed = list_resp.json()["open_loops"]
    assert listed[0]["title"] == "Fix sandbox path"

    close_resp = client.post(
        f"/api/sessions/{session_id}/open_loops/{loop_id}/close"
    )
    assert close_resp.status_code == 200
    assert close_resp.json()["open_loops"] == []

    checkpoint = checkpoints.load_latest(session_id, base_dir=checkpoint_dir)
    assert checkpoint is not None
    assert checkpoint.state.open_loops == []


def test_admin_open_loops_run_endpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    checkpoint_dir = data_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    session_id = "session-2"
    _write_checkpoint(checkpoint_dir / f"{session_id}.json", session_id)

    client = TestClient(create_app(data_root=data_root))

    create_resp = client.post(
        f"/api/sessions/{session_id}/open_loops",
        json={"title": "Say ping"},
    )
    loop_id = create_resp.json()["open_loops"][0]["id"]

    run_resp = client.post(
        f"/api/sessions/{session_id}/open_loops/run",
        json={"backend": "fake"},
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload["run_id"]
    open_loops = payload["open_loops"]
    assert any(loop.get("id") == loop_id for loop in open_loops)
