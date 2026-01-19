from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spectator.core.types import ChatMessage, Checkpoint, State

DEFAULT_DIR = Path("data") / "checkpoints"


def _checkpoint_path(session_id: str, base_dir: Path | None = None) -> Path:
    root = base_dir or DEFAULT_DIR
    return root / f"{session_id}.json"


def save_checkpoint(checkpoint: Checkpoint, base_dir: Path | None = None) -> Path:
    path = _checkpoint_path(checkpoint.session_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.revision += 1
    checkpoint.updated_ts = time.time()
    payload = asdict(checkpoint)
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
    return path


def _ensure_list_of_str(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"checkpoint state field '{field_name}' must be list[str]")


def _ensure_str(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise ValueError(f"checkpoint state field '{field_name}' must be str")


def _coerce_state(payload: Any) -> State:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint state must be an object")
    return State(
        goals=_ensure_list_of_str(payload.get("goals"), "goals"),
        open_loops=_ensure_list_of_str(payload.get("open_loops"), "open_loops"),
        decisions=_ensure_list_of_str(payload.get("decisions"), "decisions"),
        constraints=_ensure_list_of_str(payload.get("constraints"), "constraints"),
        episode_summary=_ensure_str(payload.get("episode_summary"), "episode_summary"),
        memory_tags=_ensure_list_of_str(payload.get("memory_tags"), "memory_tags"),
        memory_refs=_ensure_list_of_str(payload.get("memory_refs"), "memory_refs"),
        capabilities_granted=_ensure_list_of_str(
            payload.get("capabilities_granted"), "capabilities_granted"
        ),
        capabilities_pending=_ensure_list_of_str(
            payload.get("capabilities_pending"), "capabilities_pending"
        ),
    )


def _coerce_recent_messages(payload: Any) -> list[ChatMessage]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("checkpoint recent_messages must be a list")
    messages: list[ChatMessage] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("checkpoint recent_messages entries must be objects")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("checkpoint recent_messages entries must include role/content strings")
        messages.append(ChatMessage(role=role, content=content))
    return messages


def _coerce_trace_tail(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return payload
    raise ValueError("checkpoint trace_tail must be list[str]")


def load_latest(session_id: str, base_dir: Path | None = None) -> Checkpoint | None:
    path = _checkpoint_path(session_id, base_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be an object")
    if not isinstance(payload.get("session_id"), str):
        raise ValueError("checkpoint session_id must be str")
    if not isinstance(payload.get("revision"), int):
        raise ValueError("checkpoint revision must be int")
    if not isinstance(payload.get("updated_ts"), (float, int)):
        raise ValueError("checkpoint updated_ts must be float")
    state = _coerce_state(payload.get("state"))
    messages = _coerce_recent_messages(payload.get("recent_messages"))
    trace_tail = _coerce_trace_tail(payload.get("trace_tail"))
    return Checkpoint(
        session_id=payload["session_id"],
        revision=payload["revision"],
        updated_ts=payload["updated_ts"],
        state=state,
        recent_messages=messages,
        trace_tail=trace_tail,
    )


def load_or_create(session_id: str, base_dir: Path | None = None) -> Checkpoint:
    checkpoint = load_latest(session_id, base_dir)
    if checkpoint is not None:
        return checkpoint
    return Checkpoint(
        session_id=session_id,
        revision=0,
        updated_ts=time.time(),
        state=State(),
        recent_messages=[],
        trace_tail=[],
    )
