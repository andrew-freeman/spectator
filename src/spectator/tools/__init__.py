# src/spectator/tools/__init__.py
"""
Tools package.

Keep this module import-light to avoid circular imports.
Do NOT import ToolExecutor / runtime pipeline here.
"""

from __future__ import annotations

__all__ = [
    "ToolSettings",
    "default_tool_settings",
    "build_default_registry",
    "build_readonly_registry",
]

from spectator.tools.settings import ToolSettings, default_tool_settings

def build_default_registry(*args, **kwargs):
    # Local import to avoid import-time cycles
    from spectator.tools.registry import build_default_registry as _impl
    return _impl(*args, **kwargs)

def build_readonly_registry(*args, **kwargs):
    # Local import to avoid import-time cycles
    from spectator.tools.registry import build_readonly_registry as _impl
    return _impl(*args, **kwargs)
