# app/core/tool_registry.py
from __future__ import annotations

from typing import Set

# Read-only tools the planner is allowed to call in WORLD_QUERY mode.
READ_TOOLS: Set[str] = {
    "read_sensors",
    "read_gpu_temps",
    "read_state",
    "query_memory",
    "run_system_command",
    "read_gpu_memory",
}

# Control tools the planner is allowed to call in WORLD_CONTROL mode.
CONTROL_TOOLS: Set[str] = {
    "set_fan_speed",
    "update_state",
    "append_memory",
    "noop_control",
}

ALL_TOOLS: Set[str] = READ_TOOLS | CONTROL_TOOLS

__all__ = ["READ_TOOLS", "CONTROL_TOOLS", "ALL_TOOLS"]