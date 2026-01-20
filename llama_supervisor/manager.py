from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llama_supervisor.models import resolve_model_path
from llama_supervisor.ports import is_port_available


@dataclass(slots=True)
class ServerRecord:
    server_id: str
    status: str
    pid: int | None
    gpu: int | None
    host: str
    port: int
    model: str
    ngl: int
    verbose: bool
    started_at: float
    stopped_at: float | None
    log_path: str
    metrics_path: str


class ServerManager:
    def __init__(
        self,
        *,
        model_root: Path,
        data_root: Path,
        log_max_bytes: int,
        log_backups: int,
    ) -> None:
        self.model_root = model_root
        self.data_root = data_root
        self.log_max_bytes = log_max_bytes
        self.log_backups = log_backups
        self._records: dict[str, ServerRecord] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._registry_path = self.data_root / "llama_servers.json"
        self._load_registry()

    def list_servers(self) -> list[ServerRecord]:
        with self._lock:
            self._refresh_status_locked()
            return list(self._records.values())

    def set_model_root(self, model_root: Path) -> None:
        self.model_root = model_root.resolve()

    def start_server(self, payload: dict[str, Any]) -> ServerRecord:
        gpu = payload.get("gpu")
        port = payload.get("port")
        host = payload.get("host", "0.0.0.0")
        model = payload.get("model")
        ngl = payload.get("ngl")
        verbose = bool(payload.get("verbose", False))
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError("port must be a valid integer")
        if gpu is not None and gpu not in {0, 1}:
            raise ValueError("gpu must be 0, 1, or null")
        if gpu is None:
            ngl = 0 if ngl is None else int(ngl)
        else:
            if ngl is None:
                raise ValueError("ngl is required for GPU mode")
            ngl = int(ngl)
        if ngl < 0:
            raise ValueError("ngl must be non-negative")
        if not is_port_available(host, port):
            raise ValueError(f"port {port} is already in use")

        model_path = resolve_model_path(self.model_root, model)
        server_id = uuid.uuid4().hex
        log_path = self.data_root / "llama_logs" / f"{server_id}.log"
        metrics_path = self.data_root / "llama_metrics" / f"{server_id}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        cmd = [
            "llama-server",
            "-m",
            str(model_path),
            "-ngl",
            str(ngl),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if verbose:
            cmd.append("--verbose")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log_writer = LogWriter(
            log_path=log_path,
            max_bytes=self.log_max_bytes,
            backups=self.log_backups,
        )
        log_writer.start(process.stdout)

        record = ServerRecord(
            server_id=server_id,
            status="running",
            pid=process.pid,
            gpu=gpu,
            host=host,
            port=port,
            model=model,
            ngl=ngl,
            verbose=verbose,
            started_at=time.time(),
            stopped_at=None,
            log_path=str(log_path),
            metrics_path=str(metrics_path),
        )
        with self._lock:
            self._records[server_id] = record
            self._processes[server_id] = process
            self._save_registry_locked()
        return record

    def stop_server(self, server_id: str, timeout: float = 5.0) -> ServerRecord:
        with self._lock:
            record = self._records.get(server_id)
        if record is None:
            raise ValueError("unknown server id")
        pid = record.pid
        if pid is None:
            raise ValueError("server has no pid")
        proc = self._processes.get(server_id)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
        else:
            _terminate_pid(pid, timeout=timeout)
        with self._lock:
            record.status = "stopped"
            record.stopped_at = time.time()
            self._save_registry_locked()
        return record

    def get_record(self, server_id: str) -> ServerRecord | None:
        with self._lock:
            self._refresh_status_locked()
            return self._records.get(server_id)

    def _refresh_status_locked(self) -> None:
        for record in self._records.values():
            if record.status != "running" or record.pid is None:
                continue
            if not _pid_alive(record.pid):
                record.status = "stopped"
                record.stopped_at = record.stopped_at or time.time()
        self._save_registry_locked()

    def _save_registry_locked(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self._records.values()]
        self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, list):
            return
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                record = ServerRecord(**item)
            except TypeError:
                continue
            self._records[record.server_id] = record


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, timeout: float) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


@dataclass(slots=True)
class LogWriter:
    log_path: Path
    max_bytes: int
    backups: int
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self, stream) -> None:
        if stream is None:
            return
        self._thread = threading.Thread(
            target=self._run,
            args=(stream,),
            daemon=True,
        )
        self._thread.start()

    def _run(self, stream) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log_path.open("a", encoding="utf-8")
        try:
            for line in iter(stream.readline, ""):
                if line == "":
                    break
                handle.write(line)
                handle.flush()
                if handle.tell() >= self.max_bytes:
                    handle.close()
                    _rotate_log(self.log_path, self.backups)
                    handle = self.log_path.open("a", encoding="utf-8")
        finally:
            handle.close()


def _rotate_log(path: Path, backups: int) -> None:
    if backups <= 0 or not path.exists():
        return
    oldest = path.with_suffix(path.suffix + f".{backups}")
    oldest.unlink(missing_ok=True)
    for idx in range(backups - 1, 0, -1):
        src = path.with_suffix(path.suffix + f".{idx}")
        dest = path.with_suffix(path.suffix + f".{idx + 1}")
        if src.exists():
            src.replace(dest)
    rotated = path.with_suffix(path.suffix + ".1")
    if path.exists():
        path.replace(rotated)
