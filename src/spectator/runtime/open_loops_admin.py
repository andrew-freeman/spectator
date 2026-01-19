from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from spectator.core.types import Checkpoint
from spectator.runtime import checkpoints

MAX_TITLE_CHARS = 200
MAX_DETAILS_CHARS = 1000
MAX_TAGS = 10
MAX_TAG_CHARS = 32
MIN_PRIORITY = 0
MAX_PRIORITY = 10
ID_PREFIX = "loop-"

_ID_RE = re.compile(r"^loop-(\d+)$")


def list_open_loops(session_id: str, data_root: Path) -> list[dict[str, Any]]:
    checkpoint = checkpoints.load_latest(session_id, base_dir=data_root / "checkpoints")
    if checkpoint is None:
        raise ValueError("session not found")
    return _parse_open_loops(checkpoint.state.open_loops)


def add_open_loop(
    session_id: str,
    title: str,
    details: str | None,
    tags: list[str] | None,
    priority: int | None,
    data_root: Path,
) -> list[dict[str, Any]]:
    checkpoint = checkpoints.load_or_create(
        session_id, base_dir=data_root / "checkpoints"
    )
    entry = _build_entry(checkpoint.state.open_loops, title, details, tags, priority)
    checkpoint.state.open_loops.append(entry)
    _save_checkpoint(checkpoint, data_root)
    return _parse_open_loops(checkpoint.state.open_loops)


def close_open_loop(
    session_id: str,
    loop_id: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    checkpoint = checkpoints.load_latest(session_id, base_dir=data_root / "checkpoints")
    if checkpoint is None:
        raise ValueError("session not found")
    updated = _remove_open_loop(checkpoint.state.open_loops, loop_id)
    checkpoint.state.open_loops = updated
    _save_checkpoint(checkpoint, data_root)
    return _parse_open_loops(checkpoint.state.open_loops)


def _save_checkpoint(checkpoint: Checkpoint, data_root: Path) -> None:
    checkpoints.save_checkpoint(checkpoint, base_dir=data_root / "checkpoints")


def _build_entry(
    existing: list[str],
    title: str,
    details: str | None,
    tags: list[str] | None,
    priority: int | None,
) -> str:
    _validate_title(title)
    details = _validate_details(details)
    tags = _validate_tags(tags)
    priority = _validate_priority(priority)
    loop_id = _next_loop_id(existing)
    payload: dict[str, Any] = {
        "id": loop_id,
        "title": title.strip(),
    }
    if details:
        payload["details"] = details
    if tags:
        payload["tags"] = tags
    if priority is not None:
        payload["priority"] = priority
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _parse_open_loops(open_loops: list[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for entry in open_loops:
        parsed_entry = _parse_entry(entry)
        parsed.append(parsed_entry)
    return parsed


def _parse_entry(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"id": None, "raw": raw}
    if not isinstance(payload, dict):
        return {"id": None, "raw": raw}
    entry = {"id": payload.get("id")}
    for key in ("title", "details", "tags", "priority"):
        if key in payload:
            entry[key] = payload[key]
    return entry


def _remove_open_loop(open_loops: list[str], loop_id: str) -> list[str]:
    if not isinstance(loop_id, str) or not loop_id:
        raise ValueError("loop_id must be a non-empty string")
    remaining: list[str] = []
    removed = False
    for entry in open_loops:
        parsed = _parse_entry(entry)
        if parsed.get("id") == loop_id and not removed:
            removed = True
            continue
        remaining.append(entry)
    if not removed:
        raise ValueError("open loop not found")
    return remaining


def _next_loop_id(open_loops: list[str]) -> str:
    max_id = 0
    for entry in open_loops:
        parsed = _parse_entry(entry)
        loop_id = parsed.get("id")
        if not isinstance(loop_id, str):
            continue
        match = _ID_RE.match(loop_id)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"{ID_PREFIX}{max_id + 1}"


def _validate_title(title: str) -> None:
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    stripped = title.strip()
    if not stripped:
        raise ValueError("title must be non-empty")
    if len(stripped) > MAX_TITLE_CHARS:
        raise ValueError("title too long")


def _validate_details(details: str | None) -> str | None:
    if details is None:
        return None
    if not isinstance(details, str):
        raise ValueError("details must be a string")
    stripped = details.strip()
    if not stripped:
        return None
    if len(stripped) > MAX_DETAILS_CHARS:
        raise ValueError("details too long")
    return stripped


def _validate_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    if not isinstance(tags, list):
        raise ValueError("tags must be a list of strings")
    if len(tags) > MAX_TAGS:
        raise ValueError("too many tags")
    cleaned: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("tags must be strings")
        stripped = tag.strip()
        if not stripped:
            continue
        if len(stripped) > MAX_TAG_CHARS:
            raise ValueError("tag too long")
        cleaned.append(stripped)
    return cleaned or None


def _validate_priority(priority: int | None) -> int | None:
    if priority is None:
        return None
    if not isinstance(priority, int):
        raise ValueError("priority must be an integer")
    if priority < MIN_PRIORITY or priority > MAX_PRIORITY:
        raise ValueError("priority out of range")
    return priority
