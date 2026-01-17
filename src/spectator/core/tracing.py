from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    ts: float
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class TraceWriter:
    def __init__(
        self, session_id: str, base_dir: Path | None = None, run_id: str | None = None
    ) -> None:
        self.session_id = session_id
        self.base_dir = base_dir or Path("data") / "traces"
        self.run_id = run_id

    @property
    def path(self) -> Path:
        if self.run_id is None:
            return self.base_dir / f"{self.session_id}.jsonl"
        return self.base_dir / f"{self.session_id}__{self.run_id}.jsonl"

    def write(self, event: TraceEvent) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(event), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return self.path
