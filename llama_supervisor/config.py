from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    model_root: Path
    data_root: Path
    snapshot_interval: float
    token: str | None
    log_max_bytes: int
    log_backups: int


def load_config() -> SupervisorConfig:
    model_root = Path(os.getenv("MODEL_ROOT", "/nvme_mod/models/")).resolve()
    data_root = Path(os.getenv("SUPERVISOR_DATA_ROOT", "data")).resolve()
    snapshot_interval = float(os.getenv("SNAPSHOT_INTERVAL", "2.0"))
    token = os.getenv("SUPERVISOR_TOKEN")
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", str(200 * 1024 * 1024)))
    log_backups = int(os.getenv("LOG_BACKUPS", "3"))
    return SupervisorConfig(
        model_root=model_root,
        data_root=data_root,
        snapshot_interval=snapshot_interval,
        token=token,
        log_max_bytes=log_max_bytes,
        log_backups=log_backups,
    )

