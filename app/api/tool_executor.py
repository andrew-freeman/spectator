"""Runtime tool execution utilities."""

from __future__ import annotations

from typing import Any, Dict, List

from app.actor.actor_runner import ToolCall

from .memory_manager import MemoryManager
from .state_manager import StateManager


class ToolExecutor:
    """Execute tool calls produced by the actor/governor."""

    def __init__(self, state_manager: StateManager, memory_manager: MemoryManager):
        self._state_manager = state_manager
        self._memory_manager = memory_manager

    def execute(self, tool_call: ToolCall) -> Dict[str, Any]:
        handler = getattr(self, f"_tool_{tool_call.tool_name}", None)
        if handler is None:
            return {"tool": tool_call.tool_name, "status": "unsupported"}
        return handler(tool_call.arguments)

    def execute_many(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        return [self.execute(call) for call in tool_calls]

    # Tool handlers -----------------------------------------------------
    def _tool_read_sensors(self, _: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state_manager.read()
        return {"tool": "read_sensors", "status": "ok", "result": state.get("sensors", {})}

    def _tool_set_fan_speed(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        value = max(0.0, min(1.0, float(arguments.get("value", 0.0))))
        state = self._state_manager.update({"fan_speed": value})
        return {"tool": "set_fan_speed", "status": "ok", "result": {"fan_speed": state["fan_speed"]}}

    def _tool_read_state(self, _: Dict[str, Any]) -> Dict[str, Any]:
        return {"tool": "read_state", "status": "ok", "result": self._state_manager.read()}

    def _tool_update_state(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        delta = arguments.get("delta", {})
        updated = self._state_manager.update(delta)
        return {"tool": "update_state", "status": "ok", "result": updated}

    def _tool_append_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        content = str(arguments.get("content", ""))
        metadata = arguments.get("metadata") or {}
        entry = self._memory_manager.append(content, metadata)
        return {"tool": "append_memory", "status": "ok", "result": {"content": entry.content, "metadata": entry.metadata}}

    def _tool_query_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        keyword = str(arguments.get("keyword", ""))
        limit = int(arguments.get("limit", 5) or 5)
        matches = self._memory_manager.query(keyword, limit=limit)
        return {
            "tool": "query_memory",
            "status": "ok",
            "result": [
                {"content": entry.content, "metadata": entry.metadata}
                for entry in matches
            ],
        }


__all__ = ["ToolExecutor"]
