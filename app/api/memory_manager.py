"""In-memory storage utilities for the cognitive system."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class MemoryEntry:
    content: str
    metadata: Dict[str, str]


class MemoryManager:
    """Simple FIFO memory buffer with lightweight tagging support."""

    def __init__(self, max_entries: int = 200):
        self._max_entries = max_entries
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)

    def append(self, content: str, metadata: Optional[Dict[str, str]] = None) -> MemoryEntry:
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self._entries.append(entry)
        return entry

    def query(self, keyword: str, limit: int = 5) -> List[MemoryEntry]:
        keyword_lower = keyword.lower()
        matches: List[MemoryEntry] = []
        for entry in reversed(self._entries):
            if keyword_lower in entry.content.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break
        return list(reversed(matches))

    def export(self) -> List[Dict[str, str]]:
        return [
            {"content": entry.content, "metadata": entry.metadata}
            for entry in self._entries
        ]


__all__ = ["MemoryManager", "MemoryEntry"]
