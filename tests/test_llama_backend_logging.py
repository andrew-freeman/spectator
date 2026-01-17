from __future__ import annotations

import json
import logging

from spectator.backends.llama_server import LlamaServerBackend


def _fake_urlopen_factory(calls, response_payload: bytes):
    class FakeResponse:
        def read(self) -> bytes:
            return response_payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request, timeout=0):
        calls.append(request)
        return FakeResponse()

    return fake_urlopen


def test_llama_backend_logging_skipped_when_unset(monkeypatch, caplog) -> None:
    monkeypatch.delenv("SPECTATOR_LLAMA_LOG_PAYLOAD", raising=False)
    monkeypatch.delenv("SPECTATOR_LLAMA_LOG_DIR", raising=False)
    calls: list[object] = []
    response_payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode(
        "utf-8"
    )
    monkeypatch.setattr(
        "spectator.backends.llama_server.urllib.request.urlopen",
        _fake_urlopen_factory(calls, response_payload),
    )

    backend = LlamaServerBackend()
    with caplog.at_level(logging.INFO, logger="spectator.backends.llama_server"):
        result = backend.complete("hello", {})

    assert result == "ok"
    assert calls
    assert not [
        record
        for record in caplog.records
        if record.name == "spectator.backends.llama_server"
    ]


def test_llama_backend_logging_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SPECTATOR_LLAMA_LOG_PAYLOAD", "1")
    monkeypatch.delenv("SPECTATOR_LLAMA_LOG_DIR", raising=False)
    calls: list[object] = []
    response_payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode(
        "utf-8"
    )
    monkeypatch.setattr(
        "spectator.backends.llama_server.urllib.request.urlopen",
        _fake_urlopen_factory(calls, response_payload),
    )

    backend = LlamaServerBackend()
    params = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.25}
    expected_payload = backend._build_payload("hello", params)
    pretty_payload = json.dumps(expected_payload, indent=2)

    with caplog.at_level(logging.INFO, logger="spectator.backends.llama_server"):
        result = backend.complete("hello", params)

    assert result == "ok"
    assert calls
    request = calls[0]
    assert request.data == json.dumps(expected_payload).encode("utf-8")
    assert any(
        pretty_payload in record.getMessage()
        for record in caplog.records
        if record.name == "spectator.backends.llama_server"
    )
