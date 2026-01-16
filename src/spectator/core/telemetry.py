from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TelemetrySnapshot:
    ts: float
    pid: int
    platform: str
    python: str
    ram_total_mb: int | None
    ram_avail_mb: int | None
    notes: dict[str, Any] = field(default_factory=dict)


def _read_meminfo_mb() -> tuple[int | None, int | None]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            meminfo = handle.read().splitlines()
    except OSError:
        return None, None

    total_kb = None
    avail_kb = None
    for line in meminfo:
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                total_kb = int(parts[1])
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                avail_kb = int(parts[1])
        if total_kb is not None and avail_kb is not None:
            break

    total_mb = total_kb // 1024 if total_kb is not None else None
    avail_mb = avail_kb // 1024 if avail_kb is not None else None
    return total_mb, avail_mb


def collect_basic_telemetry() -> TelemetrySnapshot:
    ram_total_mb = None
    ram_avail_mb = None
    if platform.system().lower() == "linux":
        ram_total_mb, ram_avail_mb = _read_meminfo_mb()

    return TelemetrySnapshot(
        ts=time.time(),
        pid=os.getpid(),
        platform=platform.platform(),
        python=platform.python_version(),
        ram_total_mb=ram_total_mb,
        ram_avail_mb=ram_avail_mb,
    )
