from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ToolSettings:
    http_allowlist_enabled: bool = False
    http_allowlist: set[str] = field(default_factory=set)
    http_cache_path: Path | None = None
    http_cache_ttl_s: float = 3600.0
    http_timeout_s: float = 10.0
    http_max_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        self.http_allowlist = {item.lower() for item in self.http_allowlist}

    def with_allowlist(self, allowlist: Iterable[str]) -> "ToolSettings":
        self.http_allowlist = {item.lower() for item in allowlist}
        return self


def default_tool_settings(root: Path) -> ToolSettings:
    return ToolSettings(http_cache_path=root / ".spectator_http_cache.sqlite")
