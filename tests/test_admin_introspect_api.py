from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from spectator.admin.app import create_app

pytestmark = pytest.mark.admin


def test_admin_introspect_endpoints(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sample = repo_root / "sample.txt"
    sample.write_text("one\ntwo\nthree\n", encoding="utf-8")

    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "traces").mkdir()

    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    responses = {"governor": ["Summary ok."]}
    monkeypatch.setenv("SPECTATOR_FAKE_ROLE_RESPONSES", json.dumps(responses))

    client = TestClient(create_app(data_root=data_root))

    list_resp = client.get("/api/introspect/list?limit=10")
    assert list_resp.status_code == 200
    assert "sample.txt" in list_resp.json()["files"]

    read_resp = client.get("/api/introspect/read?path=sample.txt&lines=2")
    assert read_resp.status_code == 200
    assert read_resp.json()["content"] == "two\nthree"

    summarize_resp = client.post(
        "/api/introspect/summarize",
        json={"path": "sample.txt", "lines": 2, "backend": "fake"},
    )
    assert summarize_resp.status_code == 200
    assert summarize_resp.json()["summary"] == "Summary ok."
