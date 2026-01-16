from __future__ import annotations

from typing import Any

import spectator.tools.http_cache as http_cache_module
from spectator.core.types import State
from spectator.runtime.tool_calls import ToolCall
from spectator.tools import ToolSettings, build_default_registry


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self._index = 0
        self.status = status
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if self._index >= len(self._body):
            return b""
        if n is None or n < 0:
            n = len(self._body) - self._index
        chunk = self._body[self._index : self._index + n]
        self._index += len(chunk)
        return chunk

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def _run_http_get(
    tmp_path,
    monkeypatch,
    *,
    capabilities: list[str],
    settings: ToolSettings,
    body: bytes = b"ok",
    headers: dict[str, str] | None = None,
) -> tuple[bool, Any]:
    calls: list[str] = []

    def fake_urlopen(_request, timeout: float) -> FakeResponse:  # noqa: ARG001
        calls.append("called")
        return FakeResponse(body, headers=headers)

    monkeypatch.setattr("spectator.tools.http_tool.urlopen", fake_urlopen)
    _registry, executor = build_default_registry(tmp_path, settings=settings)
    state = State(capabilities_granted=capabilities)
    call = ToolCall(id="t1", tool="http.get", args={"url": "https://example.com"})
    result = executor.execute_calls([call], state)[0]
    return (bool(calls), result)


def test_http_get_denies_without_capability(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(http_cache_path=tmp_path / "cache.sqlite")
    called, result = _run_http_get(tmp_path, monkeypatch, capabilities=[], settings=settings)

    assert called is False
    assert result.ok is False
    assert "denied" in (result.error or "")


def test_http_get_allows_net_without_allowlist(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(http_cache_path=tmp_path / "cache.sqlite")
    called, result = _run_http_get(tmp_path, monkeypatch, capabilities=["net"], settings=settings)

    assert called is True
    assert result.ok is True
    assert result.output["url"] == "https://example.com"


def test_http_get_allowlist_blocks_net(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(
        http_cache_path=tmp_path / "cache.sqlite",
        http_allowlist_enabled=True,
        http_allowlist={"allowed.com"},
    )
    called, result = _run_http_get(tmp_path, monkeypatch, capabilities=["net"], settings=settings)

    assert called is False
    assert result.ok is False


def test_http_get_allowlist_allows_domain_cap(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(
        http_cache_path=tmp_path / "cache.sqlite",
        http_allowlist_enabled=True,
        http_allowlist={"allowed.com"},
    )
    called, result = _run_http_get(
        tmp_path,
        monkeypatch,
        capabilities=["net:example.com"],
        settings=settings,
    )

    assert called is True
    assert result.ok is True


def test_http_get_cache_hit_and_ttl_expiry(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(
        http_cache_path=tmp_path / "cache.sqlite",
        http_cache_ttl_s=0.0,
    )

    calls: list[int] = []

    def fake_urlopen(_request, timeout: float) -> FakeResponse:  # noqa: ARG001
        calls.append(1)
        return FakeResponse(b"cached")

    monkeypatch.setattr("spectator.tools.http_tool.urlopen", fake_urlopen)
    times = iter([1000.0, 1001.0, 1002.0])
    monkeypatch.setattr(http_cache_module.time, "time", lambda: next(times))
    _registry, executor = build_default_registry(tmp_path, settings=settings)
    state = State(capabilities_granted=["net"])
    call = ToolCall(id="t1", tool="http.get", args={"url": "https://example.com"})

    first = executor.execute_calls([call], state)[0]
    second = executor.execute_calls([call], state)[0]

    assert first.ok is True
    assert second.ok is True
    assert first.output["cache_hit"] is False
    assert second.output["cache_hit"] is False
    assert len(calls) == 2


def test_http_get_cache_hit_uses_cached_entry(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(http_cache_path=tmp_path / "cache.sqlite")
    calls: list[int] = []

    def fake_urlopen(_request, timeout: float) -> FakeResponse:  # noqa: ARG001
        calls.append(1)
        return FakeResponse(b"hello")

    monkeypatch.setattr("spectator.tools.http_tool.urlopen", fake_urlopen)
    _registry, executor = build_default_registry(tmp_path, settings=settings)
    state = State(capabilities_granted=["net"])
    call = ToolCall(id="t1", tool="http.get", args={"url": "https://example.com"})

    first = executor.execute_calls([call], state)[0]
    second = executor.execute_calls([call], state)[0]

    assert first.ok is True
    assert second.ok is True
    assert first.output["cache_hit"] is False
    assert second.output["cache_hit"] is True
    assert len(calls) == 1


def test_http_get_enforces_byte_cap(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(
        http_cache_path=tmp_path / "cache.sqlite",
        http_max_bytes=5,
    )
    called, result = _run_http_get(
        tmp_path,
        monkeypatch,
        capabilities=["net"],
        settings=settings,
        body=b"too-large",
    )

    assert called is True
    assert result.ok is False
    assert "byte limit" in (result.error or "")


def test_http_get_strips_html(tmp_path, monkeypatch) -> None:
    settings = ToolSettings(http_cache_path=tmp_path / "cache.sqlite")
    _, result = _run_http_get(
        tmp_path,
        monkeypatch,
        capabilities=["net"],
        settings=settings,
        body=b"<html><body>Hello <b>world</b></body></html>",
        headers={"Content-Type": "text/html"},
    )

    assert result.ok is True
    assert result.output["text"] == "Hello world"
