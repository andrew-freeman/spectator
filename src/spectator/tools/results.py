from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from spectator.core.types import ChatMessage


@dataclass(slots=True)
class ToolResult:
    id: str
    tool: str
    ok: bool
    output: Any | None
    error: str | None
    metadata: dict[str, Any] | None = None

    def to_tool_message(self) -> ChatMessage:
        payload = {
            "id": self.id,
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }
        return ChatMessage(role="tool", content=json.dumps(payload))
