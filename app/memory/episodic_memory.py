"""Persistent episodic memory store for cycle summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class EpisodicMemory:
    """Append-only JSONL store capturing the agent's lived experience."""

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append_episode(self, episode: Dict[str, Any]) -> None:
        """Append a single episode to the JSONL file."""

        payload = dict(episode)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def tail(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` episodes (newest first)."""

        if limit <= 0 or not self._path.exists():
            return []

        lines: List[str] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
        tail_lines = lines[-limit:]
        episodes: List[Dict[str, Any]] = []
        for entry in reversed(tail_lines):
            try:
                episodes.append(json.loads(entry))
            except json.JSONDecodeError:
                continue
        return episodes


__all__ = ["EpisodicMemory"]
