from __future__ import annotations

from pathlib import Path
from typing import Any

from spectator.runtime.tool_calls import ToolCall
from spectator.tools.registry import ToolRegistry
from spectator.tools.results import ToolResult


class ToolExecutor:
    def __init__(self, root: Path, registry: ToolRegistry) -> None:
        self._root = root
        self._registry = registry

    def execute_calls(self, calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            spec = self._registry.get(call.tool)
            if spec is None:
                results.append(
                    ToolResult(
                        id=call.id,
                        tool=call.tool,
                        ok=False,
                        output=None,
                        error="unknown tool",
                    )
                )
                continue
            try:
                output = spec.handler(call.args)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ToolResult(
                        id=call.id,
                        tool=call.tool,
                        ok=False,
                        output=None,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                ToolResult(
                    id=call.id,
                    tool=call.tool,
                    ok=True,
                    output=output,
                    error=None,
                )
            )
        return results
