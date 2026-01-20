from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TelemetrySnapshot:
    ts: float
    server_id: str
    pid: int
    gpu: int | None
    system: dict[str, Any]
    process: dict[str, Any]
    gpu_metrics: dict[str, Any] | None


class TelemetryCollector:
    def __init__(self, interval: float, data_root: Path, manager) -> None:
        self.interval = interval
        self.data_root = data_root
        self.manager = manager
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_cpu: dict[int, tuple[float, float]] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._collect_once()
            self._stop.wait(self.interval)

    def _collect_once(self) -> None:
        for record in self.manager.list_servers():
            if record.status != "running" or record.pid is None:
                continue
            snapshot = build_snapshot(
                server_id=record.server_id,
                pid=record.pid,
                gpu=record.gpu,
                last_cpu=self._last_cpu,
            )
            path = self.data_root / "llama_metrics" / f"{record.server_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(snapshot), ensure_ascii=True) + "\n")


def build_snapshot(
    *,
    server_id: str,
    pid: int,
    gpu: int | None,
    last_cpu: dict[int, tuple[float, float]],
) -> TelemetrySnapshot:
    system = _read_system_metrics()
    process = _read_process_metrics(pid, last_cpu)
    gpu_metrics = _read_gpu_metrics(gpu) if gpu is not None else None
    return TelemetrySnapshot(
        ts=time.time(),
        server_id=server_id,
        pid=pid,
        gpu=gpu,
        system=system,
        process=process,
        gpu_metrics=gpu_metrics,
    )


def _read_system_metrics() -> dict[str, Any]:
    loadavg = os.getloadavg()
    mem_total = None
    mem_available = None
    meminfo = _read_meminfo()
    if meminfo:
        mem_total = meminfo.get("MemTotal")
        mem_available = meminfo.get("MemAvailable")
    return {
        "loadavg": loadavg,
        "mem_total_kb": mem_total,
        "mem_available_kb": mem_available,
    }


def _read_meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    data: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            key = parts[0][:-1]
            try:
                data[key] = int(parts[1])
            except ValueError:
                continue
    return data


def _read_process_metrics(pid: int, last_cpu: dict[int, tuple[float, float]]) -> dict[str, Any]:
    rss_kb = None
    threads = None
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = _parse_kb(line)
            elif line.startswith("Threads:"):
                threads = _parse_int(line)
    cpu_percent = _compute_cpu_percent(pid, last_cpu)
    return {
        "rss_kb": rss_kb,
        "cpu_percent": cpu_percent,
        "threads": threads,
    }


def _compute_cpu_percent(pid: int, last_cpu: dict[int, tuple[float, float]]) -> float | None:
    stat_path = Path(f"/proc/{pid}/stat")
    total_path = Path("/proc/stat")
    if not stat_path.exists() or not total_path.exists():
        return None
    try:
        stat_fields = stat_path.read_text(encoding="utf-8").split()
        total_fields = total_path.read_text(encoding="utf-8").split()
        utime = float(stat_fields[13])
        stime = float(stat_fields[14])
        total_time = sum(float(value) for value in total_fields[1:8])
    except (IndexError, ValueError):
        return None
    proc_total = utime + stime
    now = time.monotonic()
    prev = last_cpu.get(pid)
    last_cpu[pid] = (proc_total, now)
    if prev is None:
        return None
    prev_total, prev_ts = prev
    elapsed = now - prev_ts
    if elapsed <= 0:
        return None
    delta_proc = proc_total - prev_total
    ticks = os.sysconf("SC_CLK_TCK")
    cpu_seconds = delta_proc / float(ticks)
    cores = os.cpu_count() or 1
    return (cpu_seconds / elapsed) * 100.0 / cores


def _read_gpu_metrics(gpu: int) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "memory_total_mb": int(float(parts[0])),
            "memory_used_mb": int(float(parts[1])),
            "memory_free_mb": int(float(parts[2])),
            "utilization_gpu": float(parts[3]),
        }
    except ValueError:
        return None


def _parse_kb(line: str) -> int | None:
    parts = line.split()
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _parse_int(line: str) -> int | None:
    parts = line.split()
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None
