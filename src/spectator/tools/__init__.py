from __future__ import annotations

from pathlib import Path

from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler, read_text_handler, write_text_handler
from spectator.tools.registry import ToolRegistry
from spectator.tools.results import ToolResult
from spectator.tools.sandbox import resolve_under_root, validate_shell_cmd
from spectator.tools.shell_tool import shell_exec_handler


def build_default_registry(root: Path) -> tuple[ToolRegistry, ToolExecutor]:
    registry = ToolRegistry()
    registry.register("fs.read_text", read_text_handler(root))
    registry.register("fs.list_dir", list_dir_handler(root))
    registry.register("fs.write_text", write_text_handler(root))
    registry.register("shell.exec", shell_exec_handler(root))
    executor = ToolExecutor(root, registry)
    return registry, executor


__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "resolve_under_root",
    "validate_shell_cmd",
]
