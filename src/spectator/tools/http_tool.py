from __future__ import annotations

import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from spectator.tools.context import ToolContext
from spectator.tools.http_cache import HttpCache
from spectator.tools.settings import ToolSettings


@dataclass(slots=True)
class HttpResponse:
    url: str
    status: int
    text: str
    cache_hit: bool


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        text = " ".join("".join(self._parts).split())
        return text.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLStripper()
    parser.feed(html)
    return parser.get_text()


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must be http or https")
    if not parsed.hostname:
        raise ValueError("url must include a hostname")
    return parsed.hostname.lower()


def _is_allowed(domain: str, capabilities: list[str], settings: ToolSettings) -> bool:
    domain_cap = f"net:{domain}"
    if domain_cap in capabilities:
        return True
    if "net" not in capabilities:
        return False
    if settings.http_allowlist_enabled:
        return domain in settings.http_allowlist
    return True


def _read_limited(response: Any, timeout_s: float, max_bytes: int) -> bytes:
    start = time.monotonic()
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("response exceeded time limit")
        chunk = response.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("response exceeded byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def http_get_handler(settings: ToolSettings) -> Any:
    cache_path = settings.http_cache_path
    cache = HttpCache(cache_path, settings.http_cache_ttl_s) if cache_path else None

    def handler(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        url = args.get("url")
        use_cache = args.get("use_cache", True)
        if not isinstance(url, str):
            raise ValueError("url must be a string")
        if not isinstance(use_cache, bool):
            raise ValueError("use_cache must be a boolean")

        domain = _extract_domain(url)
        if not _is_allowed(domain, context.state.capabilities_granted, settings):
            raise ValueError("network access denied")

        if use_cache and cache is not None:
            cached = cache.get(url)
            if cached is not None:
                return {
                    "url": url,
                    "status": cached.status,
                    "text": cached.text,
                    "cache_hit": True,
                }

        request = Request(url, headers={"User-Agent": "spectator/1.0"})
        with urlopen(request, timeout=settings.http_timeout_s) as response:
            status = int(getattr(response, "status", 200))
            content_type = ""
            if hasattr(response, "headers") and response.headers is not None:
                content_type = response.headers.get("Content-Type", "")
            if not content_type and hasattr(response, "getheader"):
                content_type = response.getheader("Content-Type") or ""
            body = _read_limited(response, settings.http_timeout_s, settings.http_max_bytes)
            text = body.decode("utf-8", errors="replace")
            if "html" in content_type.lower() or "<html" in text.lower():
                text = _html_to_text(text)

        if use_cache and cache is not None:
            cache.set(url, status, text)

        return {
            "url": url,
            "status": status,
            "text": text,
            "cache_hit": False,
        }

    return handler
