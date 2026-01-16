from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

START_MARKER = "<<<NOTES_JSON>>>"
END_MARKER = "<<<END_NOTES_JSON>>>"


@dataclass(slots=True)
class NotesPatch:
    set_goals: list[str] = field(default_factory=list)
    add_open_loops: list[str] = field(default_factory=list)
    close_open_loops: list[str] = field(default_factory=list)
    add_decisions: list[str] = field(default_factory=list)
    add_constraints: list[str] = field(default_factory=list)
    set_episode_summary: str | None = None
    add_memory_tags: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def _ensure_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def _extract_block(text: str) -> tuple[str | None, int, int]:
    start_index = text.find(START_MARKER)
    if start_index == -1:
        return None, -1, -1
    end_index = text.find(END_MARKER, start_index)
    if end_index == -1:
        return None, -1, -1
    payload_start = start_index + len(START_MARKER)
    payload = text[payload_start:end_index].strip()
    return payload, start_index, end_index + len(END_MARKER)


def _coerce_patch(data: dict[str, Any]) -> NotesPatch | None:
    set_goals = _ensure_list(data.get("set_goals"))
    add_open_loops = _ensure_list(data.get("add_open_loops"))
    close_open_loops = _ensure_list(data.get("close_open_loops"))
    add_decisions = _ensure_list(data.get("add_decisions"))
    add_constraints = _ensure_list(data.get("add_constraints"))
    add_memory_tags = _ensure_list(data.get("add_memory_tags"))
    actions = _ensure_list(data.get("actions"))
    set_episode_summary = _ensure_str(data.get("set_episode_summary"))

    if any(
        value is None
        for value in (
            set_goals,
            add_open_loops,
            close_open_loops,
            add_decisions,
            add_constraints,
            add_memory_tags,
            actions,
        )
    ):
        return None

    return NotesPatch(
        set_goals=set_goals or [],
        add_open_loops=add_open_loops or [],
        close_open_loops=close_open_loops or [],
        add_decisions=add_decisions or [],
        add_constraints=add_constraints or [],
        set_episode_summary=set_episode_summary,
        add_memory_tags=add_memory_tags or [],
        actions=actions or [],
    )


def extract_notes(text: str) -> tuple[str, NotesPatch | None]:
    payload, start_index, end_index = _extract_block(text)
    if payload is None:
        return text, None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return text, None

    if not isinstance(data, dict):
        return text, None

    patch = _coerce_patch(data)
    if patch is None:
        return text, None

    visible_text = f"{text[:start_index]}{text[end_index:]}"
    return visible_text, patch
