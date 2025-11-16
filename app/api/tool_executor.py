"""Runtime tool execution utilities for Spectator v2."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, List, Sequence

from app.core.schemas import ToolCall, ToolResult

from .memory_manager import MemoryManager
from .state_manager import StateManager

LOGGER = logging.getLogger(__name__)


class ToolExecutor:
    """Execute tool calls produced by the planner/governor."""

    def __init__(
        self,
        *,
        state_manager: StateManager,
        memory_manager: MemoryManager,
        policy_config: Dict[str, Any] | None,
        system_limits: Dict[str, Any] | None,
    ) -> None:
        self._state_manager = state_manager
        self._memory_manager = memory_manager
        self._policy_config = policy_config or {}
        self._system_limits = system_limits or {}
        self._recent_sensors: Dict[str, Dict[str, Any]] = {}
        bounds = (self._system_limits.get("fan_speed_bounds") or {}) if isinstance(self._system_limits, dict) else {}
        self._fan_min = float(bounds.get("min", 0.0))
        self._fan_max = float(bounds.get("max", 85.0))

    # Public API ---------------------------------------------------------
    def execute(self, call: ToolCall) -> ToolResult:
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            return ToolResult(tool=call.name, status="error", result={}, error=f"Unknown tool: {call.name}")
        try:
            return handler(**call.arguments)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Tool %s failed", call.name)
            return ToolResult(tool=call.name, status="error", result={}, error=str(exc))

    def execute_many(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        return [self.execute(call) for call in calls]

    # Tool handlers ------------------------------------------------------
    def _tool_read_gpu_temps(self, **_: Any) -> ToolResult:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                encoding="utf-8",
            )
            temps = [int(line.strip()) for line in output.splitlines() if line.strip()]
            self._record_sensor("gpu_temps", temps)
            return ToolResult(tool="read_gpu_temps", status="ok", result={"gpu_temps": temps})
        except Exception as exc:  # pragma: no cover - system dependency
            return ToolResult(tool="read_gpu_temps", status="error", result={}, error=str(exc))

    def _tool_set_fan_speed(self, *, fan_id: str = "gpu", speed: float, reason: str, **_: Any) -> ToolResult:
        if not reason:
            return ToolResult(tool="set_fan_speed", status="error", result={}, error="A justification reason is required.")
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            return ToolResult(tool="set_fan_speed", status="error", result={}, error="Invalid fan speed value.")
        if speed_value < self._fan_min or speed_value > self._fan_max:
            return ToolResult(
                tool="set_fan_speed",
                status="error",
                result={},
                error=f"Fan speed must be between {self._fan_min} and {self._fan_max}.",
            )
        thermal_policy = self._policy_config.get("thermal_policy", {})
        target_max = thermal_policy.get("target_max")
        if target_max is not None:
            recent = self._recent_sensors.get("gpu_temps")
            if recent and isinstance(recent.get("value"), list):
                if all(temp < target_max - 5 for temp in recent["value"]):
                    return ToolResult(
                        tool="set_fan_speed",
                        status="error",
                        result={},
                        error="Temperatures already below target; control action unnecessary.",
                    )
        updated = self._state_manager.update(
            {
                "fan_speed": speed_value,
                "fan_id": fan_id,
                "fan_reason": reason,
                "fan_timestamp": time.time(),
            }
        )
        return ToolResult(
            tool="set_fan_speed",
            status="ok",
            result={
                "fan_speed": updated.get("fan_speed"),
                "fan_id": fan_id,
                "reason": reason,
            },
        )

    def _tool_read_state(self, **_: Any) -> ToolResult:
        return ToolResult(tool="read_state", status="ok", result=self._state_manager.read())

    def _tool_read_sensors(self, **_: Any) -> ToolResult:
        sensors = self._state_manager.read().get("sensors") or {}
        return ToolResult(tool="read_sensors", status="ok", result=sensors)

    # Helpers ------------------------------------------------------------
    def _record_sensor(self, name: str, value: Any) -> None:
        self._recent_sensors[name] = {"value": value, "timestamp": time.time()}

    # Legacy compatibility -----------------------------------------------
    def _read_gpu_temps(self) -> Dict[str, Any]:
        """Expose a simple helper used by legacy UI components."""
        result = self._tool_read_gpu_temps()
        if result.status == "ok":
            return result.result
        return {"error": result.error or "unavailable"}


__all__ = ["ToolExecutor"]
