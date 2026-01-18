from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple, Optional

ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ToolSpec:
    name: str
    handler: ToolHandler


def build_default_registry(
    root: Path,
    *,
    settings=None,
) -> tuple["ToolRegistry", "ToolExecutor"]:
    """
    Construct the default tool registry + executor.

    Keep imports inside the function to avoid circular import issues.
    """
    # Local imports to avoid import-time cycles
    from spectator.tools.executor import ToolExecutor
    from spectator.tools.settings import ToolSettings, default_tool_settings
    from spectator.tools.fs_tools import read_text_handler, write_text_handler, list_dir_handler
    from spectator.tools.shell_tool import shell_exec_handler

    # http tool is optional depending on your repo state; import lazily
    try:
        from spectator.tools.http_tool import http_get_handler  # rename if your handler differs
    except Exception:
        http_get_handler = None

    if settings is None:
        settings = default_tool_settings(root)
    elif not isinstance(settings, ToolSettings):
        raise ValueError("settings must be a ToolSettings instance")

    reg = ToolRegistry()

    # FS tools
    reg.register("fs.read_text", read_text_handler(root))
    reg.register("fs.write_text", write_text_handler(root))
    reg.register("fs.list_dir", list_dir_handler(root))

    # Shell tool
    reg.register("shell.exec", shell_exec_handler(root))

    # HTTP tool (only if available)
    if http_get_handler is not None:
        reg.register("http.get", http_get_handler(root, settings=settings))

    executor = ToolExecutor(reg)
    return reg, executor

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._tools[name] = ToolSpec(name=name, handler=handler)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())
