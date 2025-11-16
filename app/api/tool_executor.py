"""Runtime tool execution utilities."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.actor.actor_runner import ToolCall

from .memory_manager import MemoryManager
from .state_manager import StateManager


DEFAULT_TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "read_gpu_temps": {
        "kind": "sensor",
        "description": "Read GPU temperatures using nvidia-smi",
        "side_effect": "none",
        "max_frequency_hz": 2,
    },
    "set_fan_speed": {
        "kind": "actuator",
        "description": "Set system fan speed in percent",
        "side_effect": "moderate",
        "params": {
            "speed": {"type": "float", "min": 0.0, "max": 80.0},
        },
        "safety": {
            "require_reason": True,
            "require_recent_reading_s": 30,
        },
    },
    "read_sensors": {"kind": "sensor", "description": "Read cached sensor data.", "side_effect": "none"},
    "read_state": {"kind": "sensor", "description": "Read shared state", "side_effect": "none"},
    "update_state": {"kind": "actuator", "description": "Update shared state", "side_effect": "low"},
    "append_memory": {"kind": "actuator", "description": "Write to memory", "side_effect": "low"},
    "query_memory": {"kind": "sensor", "description": "Read memory", "side_effect": "none"},
    "run_system_command": {"kind": "sensor", "description": "Run whitelisted commands", "side_effect": "moderate"},
    "who_are_you": {"kind": "service", "description": "Describe identity", "side_effect": "none"},
}

_TOOL_SPEC_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


class ToolExecutor:
    """Execute tool calls produced by the actor/governor."""

    def __init__(
        self,
        state_manager: StateManager,
        memory_manager: MemoryManager,
        *,
        tool_config_path: Optional[Path] = None,
    ):
        self._state_manager = state_manager
        self._memory_manager = memory_manager
        self._tool_specs = _load_tool_specs(tool_config_path)
        self._last_call_time: Dict[str, float] = {}
        self._recent_readings: Dict[str, Dict[str, Any]] = {}

    def _read_gpu_temps(self) -> Dict[str, Any]:
        import subprocess

        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                encoding="utf-8",
            )
            temps = [int(x.strip()) for x in output.splitlines() if x.strip().isdigit()]
            return {"gpu_temps": temps}
        except Exception as exc:  # pragma: no cover - system dependency
            LOGGER.exception("Failed to read GPU temperatures")
            return {"error": str(exc)}

    def execute(self, tool_call: ToolCall) -> Dict[str, Any]:
        handler = getattr(self, f"_tool_{tool_call.tool_name}", None)
        spec = self._tool_specs.get(tool_call.tool_name)
        if handler is None or spec is None:
            return _error_payload(tool_call.tool_name, f"Unknown tool: {tool_call.tool_name}")

        arguments = dict(tool_call.arguments or {})
        error = self._validate_tool_call(tool_call.tool_name, arguments, spec)
        if error:
            return _error_payload(tool_call.tool_name, error)

        result = handler(arguments)
        self._last_call_time[tool_call.tool_name] = time.monotonic()
        return result

    def execute_many(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for call in tool_calls:
            results.append(self.execute(call))
        return results

    # Tool handlers -----------------------------------------------------
    def _tool_read_sensors(self, _: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state_manager.read()
        return {"tool": "read_sensors", "status": "ok", "result": state.get("sensors", {})}

    def _tool_set_fan_speed(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        speed = float(arguments.get("speed", 0.0))
        reason = str(arguments.get("reason", ""))
        state = self._state_manager.update(
            {
                "fan_speed": speed,
                "fan_speed_reason": reason,
                "fan_speed_timestamp": time.time(),
            }
        )
        return {
            "tool": "set_fan_speed",
            "status": "ok",
            "result": {
                "fan_speed": state["fan_speed"],
                "reason": state.get("fan_speed_reason", ""),
            },
        }

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

    def _tool_read_gpu_temps(self, _: Dict[str, Any]) -> Dict[str, Any]:
        result = self._read_gpu_temps()
        status = "ok" if "gpu_temps" in result else "error"
        if status == "ok":
            self._record_sensor("gpu_temps", result["gpu_temps"])
        return {"tool": "read_gpu_temps", "status": status, "result": result}

    def _tool_run_system_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self._run_system_command(arguments)
        status = "ok" if "stdout" in result else "error"
        return {"tool": "run_system_command", "status": status, "result": result}

    def _run_system_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        import shlex
        import subprocess

        cmd = arguments.get("cmd")
        if not cmd:
            return {"error": "No command provided"}

        allowed = ["nvidia-smi", "sensors", "uptime"]
        command_name = shlex.split(cmd)[0] if cmd else ""
        if command_name not in allowed:
            return {"error": "Command not allowed"}

        try:
            output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT).decode()
            return {"stdout": output}
        except Exception as exc:  # pragma: no cover - defensive system call guard
            return {"error": str(exc)}

    def _tool_who_are_you(self, _: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool": "who_are_you",
            "status": "ok",
            "result": "I am Spectator, your reflection-driven autonomous reasoning system.",
        }

    def get_recent_readings(self) -> Dict[str, Any]:
        return {name: data.get("value") for name, data in self._recent_readings.items()}

    # Internal helpers ---------------------------------------------------
    def _validate_tool_call(self, tool_name: str, arguments: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
        if "speed" not in arguments and "value" in arguments:
            arguments["speed"] = arguments.get("value")

        freq = spec.get("max_frequency_hz")
        if freq:
            min_interval = 1.0 / float(freq)
            last = self._last_call_time.get(tool_name)
            if last and time.monotonic() - last < min_interval:
                return f"{tool_name} called too frequently; wait before retrying."

        params = spec.get("params") or {}
        for name, meta in params.items():
            if name not in arguments:
                return f"Missing parameter '{name}' for {tool_name}."
            value = arguments[name]
            try:
                if meta.get("type") == "float":
                    value = float(value)
                elif meta.get("type") == "int":
                    value = int(value)
            except (TypeError, ValueError):
                return f"Parameter '{name}' must be a valid {meta.get('type', 'value')}"
            min_value = meta.get("min")
            max_value = meta.get("max")
            if min_value is not None and value < float(min_value):
                return f"Parameter '{name}' must be >= {min_value}."
            if max_value is not None and value > float(max_value):
                return f"Parameter '{name}' must be <= {max_value}."
            arguments[name] = value

        safety = spec.get("safety") or {}
        if spec.get("kind") == "actuator" and safety.get("require_reason"):
            reason = str(arguments.get("reason", "")).strip()
            if not reason:
                return "Actuator commands require a justification reason."
        recent_window = safety.get("require_recent_reading_s")
        if recent_window:
            if not self._has_recent_sensor("gpu_temps", float(recent_window)):
                return "Need a recent GPU temperature reading before issuing this command."
        return None

    def _record_sensor(self, name: str, value: Any) -> None:
        self._recent_readings[name] = {"value": value, "timestamp": time.monotonic()}

    def _has_recent_sensor(self, name: str, max_age: float) -> bool:
        reading = self._recent_readings.get(name)
        if not reading:
            return False
        return time.monotonic() - reading["timestamp"] <= max_age


LOGGER = logging.getLogger(__name__)


__all__ = ["ToolExecutor"]


def _load_tool_specs(tool_config_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    cache_key = str(tool_config_path) if tool_config_path else "__default__"
    if cache_key in _TOOL_SPEC_CACHE:
        return copy.deepcopy(_TOOL_SPEC_CACHE[cache_key])

    specs = copy.deepcopy(DEFAULT_TOOL_SPECS)
    if tool_config_path and tool_config_path.exists():
        try:
            with tool_config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            for name, definition in (loaded.get("tools") or {}).items():
                specs[name] = definition
        except yaml.YAMLError:
            LOGGER.warning("Failed to parse %s; falling back to defaults", tool_config_path)

    _TOOL_SPEC_CACHE[cache_key] = specs
    return copy.deepcopy(specs)


def _error_payload(tool: str, message: str) -> Dict[str, Any]:
    return {"tool": tool, "status": "error", "error": message}
