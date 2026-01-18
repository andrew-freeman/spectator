from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectator.admin.app import create_app

pytestmark = pytest.mark.admin


def test_admin_run_turn_creates_trace(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    responses = {
        "reflection": ["Reflection"],
        "planner": ["Plan"],
        "critic": ["Critic"],
        "governor": ["Final response"],
    }
    monkeypatch.setenv("SPECTATOR_FAKE_ROLE_RESPONSES", json.dumps(responses))

    client = TestClient(create_app(data_root=data_root))

    payload = {"session_id": "session-2", "text": "Hello", "backend": "fake"}
    response = client.post("/api/run_turn", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert "final_text" in body
    assert body["run_id"]
    trace_name = body["trace_file_name"]
    assert trace_name

    trace_path = data_root / "traces" / trace_name
    assert trace_path.exists()

    run_detail = client.get(
        f"/api/sessions/{payload['session_id']}/runs/{body['run_id']}"
    )
    assert run_detail.status_code == 200
    assert run_detail.json()["final_response"] == "Final response"
