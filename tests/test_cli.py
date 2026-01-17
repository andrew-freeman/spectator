from __future__ import annotations

import json

import pytest

from spectator import cli


def test_cli_run_backend_fake(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_complete(self, prompt: str, params: dict[str, object] | None = None) -> str:
        calls.append({"prompt": prompt, "params": params or {}})
        return "ok"

    monkeypatch.setattr("spectator.backends.fake.FakeBackend.complete", fake_complete)

    exit_code = cli.main(["run", "--backend", "fake", "--text", "Hello"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "ok"
    assert calls


def test_cli_run_backend_llama(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[object] = []
    payload = json.dumps({"choices": [{"message": {"content": "llama-ok"}}]}).encode("utf-8")

    class FakeResponse:
        def read(self) -> bytes:
            return payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request, timeout=0):
        calls.append(request)
        return FakeResponse()

    monkeypatch.setattr("spectator.backends.llama_server.urllib.request.urlopen", fake_urlopen)

    exit_code = cli.main(
        [
            "run",
            "--backend",
            "llama",
            "--llama-url",
            "http://example.com",
            "--text",
            "Hello",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "llama-ok"
    assert len(calls) >= 1


def test_cli_unknown_backend(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--backend", "unknown", "--text", "Hello"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "invalid choice" in captured.err
