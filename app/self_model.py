from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class SelfModel:
    identity: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    open_loops: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    self_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this SelfModel."""
        return {
            "identity": dict(self.identity),
            "capabilities": list(self.capabilities),
            "open_loops": [dict(loop) for loop in self.open_loops],
            "hypotheses": list(self.hypotheses),
            "preferences": dict(self.preferences),
            "self_notes": list(self.self_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfModel":
        """Construct a SelfModel from a dict, filling in defaults for missing fields."""
        if not isinstance(data, dict):
            return default_self_model()

        return cls(
            identity=dict(data.get("identity", {})),
            capabilities=list(data.get("capabilities", [])),
            open_loops=[dict(loop) for loop in data.get("open_loops", []) if isinstance(loop, dict)],
            hypotheses=list(data.get("hypotheses", [])),
            preferences=dict(data.get("preferences", {})),
            self_notes=list(data.get("self_notes", [])),
        )


def default_self_model() -> SelfModel:
    """Return the default SelfModel configuration."""
    return SelfModel(
        identity={
            "name": "Spectator",
            "version": "v3-reflective",
            "persona": "local self-monitoring reasoning process on this workstation",
        },
        capabilities=[
            "monitor_temperatures",
            "adjust_fans_within_policy",
            "reason_in_modes",
            "perform_self_reflection",
        ],
        open_loops=[
            {
                "goal": "monitor and stabilize GPU thermal state",
                "status": "ongoing",
                "last_update_cycle": 0,
            }
        ],
        hypotheses=[],
        preferences={
            "cooling_aggressiveness": 0.5,
            "noise_tolerance": "medium",
        },
        self_notes=[],
    )


def load_self_model(path: Path) -> SelfModel:
    """Load a SelfModel from JSON file at `path`, or return a default model if not present or invalid."""
    try:
        if not path.exists():
            return default_self_model()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return SelfModel.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return default_self_model()
    return default_self_model()


def save_self_model(model: SelfModel, path: Path) -> None:
    """Persist the SelfModel as JSON to `path`, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2, ensure_ascii=False)


def _ensure_capability(model: SelfModel, capability: str) -> None:
    if capability and capability not in model.capabilities:
        model.capabilities.append(capability)


def _find_open_loop(model: SelfModel, goal: str) -> Optional[Dict[str, Any]]:
    for loop in model.open_loops:
        if loop.get("goal") == goal:
            return loop
    return None


def _append_self_note(model: SelfModel, note: str, max_self_notes: int) -> None:
    if not note:
        return
    model.self_notes.append(note)
    if len(model.self_notes) > max_self_notes:
        del model.self_notes[:-max_self_notes]


def update_self_model_from_cycle(
    model: SelfModel,
    cycle_record: Dict[str, Any],
    *,
    max_self_notes: int = 50,
) -> SelfModel:
    reflection = cycle_record.get("reflection") or {}
    governor = cycle_record.get("governor") or {}
    critic = cycle_record.get("critic") or {}
    tool_results = cycle_record.get("tool_results") or []

    cycle_id = cycle_record.get("cycle")
    reflection_goal = reflection.get("goal") if isinstance(reflection, dict) else None
    governor_verdict = governor.get("verdict") if isinstance(governor, dict) else None
    governor_mode = governor.get("mode") if isinstance(governor, dict) else None

    # Ensure thermal monitoring open loop when goals mention relevant keywords
    goal_text = (reflection_goal or "").lower()
    if any(keyword in goal_text for keyword in ("thermal", "gpu", "temperature")):
        loop = _find_open_loop(model, "monitor and stabilize GPU thermal state")
        if loop is None:
            loop = {
                "goal": "monitor and stabilize GPU thermal state",
                "status": "ongoing",
                "last_update_cycle": cycle_id or 0,
            }
            model.open_loops.append(loop)
        else:
            if cycle_id is not None:
                loop["last_update_cycle"] = cycle_id

    # Update capabilities based on tool successes
    successful_tool = None
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        if result.get("status") != "ok":
            continue
        tool_name = result.get("tool") or result.get("name")
        if tool_name == "read_gpu_temps":
            _ensure_capability(model, "monitor_temperatures")
        if tool_name == "set_fan_speed":
            _ensure_capability(model, "adjust_fans_within_policy")
        if successful_tool is None and tool_name:
            successful_tool = tool_name

    # Update hypotheses conservatively
    critic_notes = critic.get("notes") if isinstance(critic, dict) else None
    if isinstance(critic_notes, str):
        lowered = critic_notes.lower()
        if any(word in lowered for word in ("pattern", "recurring", "concern")):
            hypothesis = critic_notes.strip()
            if hypothesis and hypothesis not in model.hypotheses:
                model.hypotheses.append(hypothesis)

    # Build self-note
    parts: List[str] = []
    if cycle_id is not None:
        parts.append(f"Cycle {cycle_id}:")
    if reflection_goal:
        parts.append(f"goal={reflection_goal}")
    if governor_verdict:
        if governor_mode:
            parts.append(f"{governor_verdict} {governor_mode}")
        else:
            parts.append(governor_verdict)
    if successful_tool:
        parts.append(f"tool {successful_tool} ok")
    note = " ".join(parts)
    _append_self_note(model, note, max_self_notes)

    return model


def export_self_model_for_prompt(model: SelfModel) -> Dict[str, Any]:
    """Return a compact, sanitized dict representation of the SelfModel for prompt inclusion."""
    data = model.to_dict()
    data["self_notes"] = data.get("self_notes", [])[-10:]
    data["hypotheses"] = data.get("hypotheses", [])[-10:]
    return data


__all__ = [
    "SelfModel",
    "default_self_model",
    "load_self_model",
    "save_self_model",
    "update_self_model_from_cycle",
    "export_self_model_for_prompt",
]
