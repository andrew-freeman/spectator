from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from spectator.core.types import State
from spectator.runtime.tool_calls import ToolCall
from spectator.tools.context import ToolContext
from spectator.tools.registry import ToolRegistry
from spectator.tools.results import ToolResult
from spectator.tools.settings import ToolSettings, default_tool_settings


class ToolExecutor:
    def __init__(
        self,
        root: Path,
        registry: ToolRegistry,
        settings: ToolSettings | None = None,
    ) -> None:
        self._root = root
        self._registry = registry
        self._settings = settings or default_tool_settings(root)

    def execute_calls(self, calls: list[ToolCall], state: State) -> list[ToolResult]:
        context = ToolContext(state=state, settings=self._settings)
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
            metadata: dict[str, Any] | None = None
            if call.tool == "http.get":
                metadata = {"url": call.args.get("url"), "cache_hit": False}
            try:
                handler = spec.handler
                signature = inspect.signature(handler)
                params = signature.parameters.values()
                positional_params = [
                    param
                    for param in params
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                ]
                has_varargs = any(param.kind is param.VAR_POSITIONAL for param in params)
                if has_varargs or len(positional_params) >= 2:
                    output = handler(call.args, context)
                else:
                    output = handler(call.args)
                if call.tool == "http.get":
                    if isinstance(output, dict):
                        metadata["cache_hit"] = bool(output.get("cache_hit", False))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ToolResult(
                        id=call.id,
                        tool=call.tool,
                        ok=False,
                        output=None,
                        error=str(exc),
                        metadata=metadata
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
                    metadata=metadata,
                )
            )
        return results

    def list_tools(self) -> list[str]:
        return [spec.name for spec in self._registry.list_tools()]
