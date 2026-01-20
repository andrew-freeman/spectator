from __future__ import annotations

import io
from pathlib import Path

import pytest

import llama_supervisor.manager as manager_mod
from llama_supervisor.manager import ServerManager


class FakePopen:
    def __init__(self, args, stdout=None, stderr=None, text=None, env=None):
        self.args = args
        if stdout is None or stdout == manager_mod.subprocess.PIPE:
            self.stdout = io.StringIO("")
        else:
            self.stdout = stdout
        self.stderr = stderr
        self.text = text
        self.env = env or {}
        self.pid = 12345
        self._terminated = False

    def terminate(self):
        self._terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._terminated = True


def test_start_server_builds_command(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model.gguf"
    model.write_text("data", encoding="utf-8")
    data_root = tmp_path / "data"

    popen_calls: list[FakePopen] = []

    def fake_popen(*args, **kwargs):
        proc = FakePopen(*args, **kwargs)
        popen_calls.append(proc)
        return proc

    monkeypatch.setattr(manager_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(manager_mod, "is_port_available", lambda host, port: True)

    manager = ServerManager(
        model_root=model_root,
        data_root=data_root,
        log_max_bytes=1024,
        log_backups=1,
    )
    record = manager.start_server(
        {
            "gpu": 0,
            "port": 8080,
            "host": "0.0.0.0",
            "model": "model.gguf",
            "ngl": 99,
            "verbose": True,
        }
    )

    assert record.pid == 12345
    assert popen_calls
    call = popen_calls[0]
    assert call.args[:2] == ["llama-server", "-m"]
    assert call.args[2] == str(model.resolve())
    assert "-ngl" in call.args
    assert "--host" in call.args
    assert "--port" in call.args
    assert "--verbose" in call.args
    assert call.env.get("CUDA_VISIBLE_DEVICES") == "0"


def test_start_server_rejects_port_in_use(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model.gguf"
    model.write_text("data", encoding="utf-8")
    data_root = tmp_path / "data"

    monkeypatch.setattr(manager_mod, "is_port_available", lambda host, port: False)

    manager = ServerManager(
        model_root=model_root,
        data_root=data_root,
        log_max_bytes=1024,
        log_backups=1,
    )

    with pytest.raises(ValueError, match="already in use"):
        manager.start_server(
            {
                "gpu": 0,
                "port": 8080,
                "host": "0.0.0.0",
                "model": "model.gguf",
                "ngl": 99,
            }
        )
