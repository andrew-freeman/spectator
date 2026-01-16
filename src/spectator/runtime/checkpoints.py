from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from spectator.core.types import ChatMessage, Checkpoint, State

DEFAULT_DIR = Path("data") / "checkpoints"


def _checkpoint_path(session_id: str, base_dir: Path | None = None) -> Path:
    root = base_dir or DEFAULT_DIR
    return root / f"{session_id}.json"


def save_checkpoint(checkpoint: Checkpoint, base_dir: Path | None = None) -> Path:
    path = _checkpoint_path(checkpoint.session_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(checkpoint)
    payload["updated_ts"] = checkpoint.updated_ts or time.time()
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
    return path


def load_latest(session_id: str, base_dir: Path | None = None) -> Checkpoint | None:
    path = _checkpoint_path(session_id, base_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = State(**payload["state"])
    messages = [ChatMessage(**item) for item in payload.get("recent_messages", [])]
    return Checkpoint(
        session_id=payload["session_id"],
        revision=payload["revision"],
        updated_ts=payload["updated_ts"],
        state=state,
        recent_messages=messages,
        trace_tail=payload.get("trace_tail", []),
    )
