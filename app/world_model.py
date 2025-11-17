from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class WorldModel:
    last_state: Dict[str, Any] = field(default_factory=dict)
    gpu_temps_history: List[Dict[str, Any]] = field(default_factory=list)
    gpu_memory_history: List[Dict[str, Any]] = field(default_factory=list)
    sensor_history: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this WorldModel."""
        return {
            "last_state": dict(self.last_state),
            "gpu_temps_history": [dict(item) for item in self.gpu_temps_history],
            "gpu_memory_history": [dict(item) for item in self.gpu_memory_history],
            "sensor_history": [dict(item) for item in self.sensor_history],
            "events": [dict(item) for item in self.events],
            "anomalies": [dict(item) for item in self.anomalies],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModel":
        """Construct a WorldModel from a dict, filling in defaults for missing fields."""
        if not isinstance(data, dict):
            return default_world_model()

        def _dict_list(items: Any) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            copies: List[Dict[str, Any]] = []
            for item in items:
                if isinstance(item, dict):
                    copies.append(dict(item))
            return copies

        return cls(
            last_state=dict(data.get("last_state", {})) if isinstance(data.get("last_state"), dict) else {},
            gpu_temps_history=_dict_list(data.get("gpu_temps_history")),
            gpu_memory_history=_dict_list(data.get("gpu_memory_history")),
            sensor_history=_dict_list(data.get("sensor_history")),
            events=_dict_list(data.get("events")),
            anomalies=_dict_list(data.get("anomalies")),
        )


def default_world_model() -> WorldModel:
    """Return a WorldModel with default empty values."""
    return WorldModel()


def load_world_model(path: Path) -> WorldModel:
    """Load a WorldModel from JSON file at `path`, or return a default model if not present or invalid."""
    try:
        if not path.exists():
            return default_world_model()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return WorldModel.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return default_world_model()
    return default_world_model()


def save_world_model(model: WorldModel, path: Path) -> None:
    """Persist the WorldModel as JSON to `path`, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2, ensure_ascii=False)


def _ensure_max_entries(items: List[Dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        items.clear()
        return
    if len(items) > limit:
        del items[:-limit]


def _coerce_number_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list):
        return None
    numbers: List[float] = []
    for entry in value:
        if isinstance(entry, (int, float)):
            numbers.append(float(entry))
        else:
            return None
    return numbers


def update_world_model_from_cycle(
    model: WorldModel,
    cycle_record: Dict[str, Any],
    *,
    max_history: int = 200,
    max_events: int = 200,
    max_anomalies: int = 100,
    hot_temp_threshold: float = 80.0,
) -> WorldModel:
    """Update the world model with information from a completed reasoning cycle."""

    state = cycle_record.get("state")
    if isinstance(state, dict):
        model.last_state = dict(state)

    cycle_raw = cycle_record.get("cycle")
    cycle_id = cycle_raw if isinstance(cycle_raw, int) else None

    tool_results = cycle_record.get("tool_results")
    if not isinstance(tool_results, list):
        tool_results = []

    temps_triggered_hot = False

    for entry in tool_results:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        tool_name = entry.get("tool") or entry.get("name")
        result_payload = entry.get("result")

        if tool_name == "read_gpu_temps" and status == "ok":
            temps = entry.get("gpu_temps")
            if temps is None and isinstance(result_payload, dict):
                temps = result_payload.get("gpu_temps")
            temps_list = _coerce_number_list(temps)
            if temps_list is not None:
                model.gpu_temps_history.append({"cycle": cycle_id, "temps": temps_list})
                if any(temp > hot_temp_threshold for temp in temps_list):
                    temps_triggered_hot = True

        elif tool_name == "read_gpu_memory" and status == "ok":
            memory = entry.get("memory_used_mb")
            if memory is None and isinstance(result_payload, dict):
                memory = result_payload.get("memory_used_mb")
            memory_list = _coerce_number_list(memory)
            if memory_list is not None:
                model.gpu_memory_history.append({"cycle": cycle_id, "memory_used_mb": memory_list})

        elif tool_name == "read_sensors" and status == "ok":
            sensors = entry.get("sensors")
            if sensors is None:
                sensors = result_payload
                if isinstance(result_payload, dict) and "sensors" in result_payload:
                    sensors = result_payload.get("sensors")
            if isinstance(sensors, dict):
                model.sensor_history.append({"cycle": cycle_id, "sensors": dict(sensors)})

        if status == "error" and tool_name in {"read_gpu_temps", "set_fan_speed"}:
            model.anomalies.append(
                {
                    "cycle": cycle_id,
                    "severity": "low",
                    "reason": f"Tool {tool_name} failed",
                }
            )

        if tool_name == "set_fan_speed" and status == "ok":
            model.events.append(
                {
                    "cycle": cycle_id,
                    "kind": "control_action",
                    "message": "set_fan_speed executed successfully",
                }
            )

    if temps_triggered_hot:
        model.anomalies.append(
            {
                "cycle": cycle_id,
                "severity": "medium",
                "reason": f"GPU temperature exceeded {hot_temp_threshold}\u00b0C",
            }
        )

    governor = cycle_record.get("governor")
    governor_dict = governor if isinstance(governor, dict) else {}
    governor_verdict = governor_dict.get("verdict")

    if governor_verdict == "reject":
        model.events.append(
            {
                "cycle": cycle_id,
                "kind": "warning",
                "message": "Governor rejected the plan.",
            }
        )
    elif governor_verdict == "query_mode":
        model.events.append(
            {
                "cycle": cycle_id,
                "kind": "info",
                "message": "read-only query executed",
            }
        )

    _ensure_max_entries(model.gpu_temps_history, max_history)
    _ensure_max_entries(model.gpu_memory_history, max_history)
    _ensure_max_entries(model.sensor_history, max_history)
    _ensure_max_entries(model.events, max_events)
    _ensure_max_entries(model.anomalies, max_anomalies)

    return model


def export_world_model_for_prompt(model: WorldModel) -> Dict[str, Any]:
    """Return a compact, sanitized dict representation of the WorldModel."""
    data = model.to_dict()
    for key in (
        "gpu_temps_history",
        "gpu_memory_history",
        "sensor_history",
        "events",
        "anomalies",
    ):
        history = data.get(key)
        if isinstance(history, list) and len(history) > 10:
            data[key] = history[-10:]
    return data


__all__ = [
    "WorldModel",
    "default_world_model",
    "load_world_model",
    "save_world_model",
    "update_world_model_from_cycle",
    "export_world_model_for_prompt",
]
