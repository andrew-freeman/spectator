from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    event_type: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


class TraceWriter:
    def __init__(self, session_id: str, base_dir: Path | None = None) -> None:
        self.session_id = session_id
        self.base_dir = base_dir or Path("data") / "traces"

    @property
    def path(self) -> Path:
        return self.base_dir / f"{self.session_id}.jsonl"

    def write(self, event: TraceEvent) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(event), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return self.path
