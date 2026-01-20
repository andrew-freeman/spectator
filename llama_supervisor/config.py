from __future__ import annotations

import json
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
    config = SupervisorConfig(
        model_root=model_root,
        data_root=data_root,
        snapshot_interval=snapshot_interval,
        token=token,
        log_max_bytes=log_max_bytes,
        log_backups=log_backups,
    )
    settings = load_settings(config.data_root)
    if settings.model_root is not None:
        config = SupervisorConfig(
            model_root=settings.model_root,
            data_root=config.data_root,
            snapshot_interval=config.snapshot_interval,
            token=config.token,
            log_max_bytes=config.log_max_bytes,
            log_backups=config.log_backups,
        )
    return config


@dataclass(frozen=True, slots=True)
class SupervisorSettings:
    model_root: Path | None = None


def load_settings(data_root: Path) -> SupervisorSettings:
    path = data_root / "llama_supervisor.json"
    if not path.exists():
        return SupervisorSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return SupervisorSettings()
    if not isinstance(payload, dict):
        return SupervisorSettings()
    model_root_raw = payload.get("model_root")
    model_root = None
    if isinstance(model_root_raw, str) and model_root_raw:
        model_root = Path(model_root_raw).resolve()
    return SupervisorSettings(model_root=model_root)


def save_settings(data_root: Path, settings: SupervisorSettings) -> None:
    path = data_root / "llama_supervisor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_root": str(settings.model_root) if settings.model_root else None}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
