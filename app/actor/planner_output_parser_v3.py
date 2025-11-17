# app/actor/planner_output_parser_v3.py
from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional, Set

from app.core.tool_registry import READ_TOOLS, CONTROL_TOOLS, ALL_TOOLS


SECTION_ORDER = [
    "MODE",
    "ANALYSIS",
    "STEPS",
    "TOOL_CALLS",
    "NEEDS_RISK_CHECK",
    "CONFIDENCE",
]


class PlannerParseError(ValueError):
    """Raised when the planner output cannot be parsed according to the V3 schema."""


def _split_sections(raw: str) -> Dict[str, List[str]]:
    lines = (raw or "").splitlines()
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_ORDER}
    current: Optional[str] = None

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("#MODE:"):
            current = "MODE"
            continue
        if stripped.startswith("#ANALYSIS:"):
            current = "ANALYSIS"
            continue
        if stripped.startswith("#STEPS:"):
            current = "STEPS"
            continue
        if stripped.startswith("#TOOL_CALLS:"):
            current = "TOOL_CALLS"
            continue
        if stripped.startswith("#NEEDS_RISK_CHECK:"):
            current = "NEEDS_RISK_CHECK"
            continue
        if stripped.startswith("#CONFIDENCE:"):
            current = "CONFIDENCE"
            continue

        if current is not None:
            sections[current].append(stripped)

    return sections


def _parse_mode(lines: List[str], fallback: str = "chat") -> str:
    for line in lines:
        val = line.strip().lower()
        if not val:
            continue
        if val in {"chat", "knowledge", "world_query", "world_control"}:
            return val
        raise PlannerParseError(f"Invalid mode value: {val!r}")
    return fallback


def _parse_analysis(lines: List[str], default: str = "") -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return text or default


def _parse_steps(lines: List[str]) -> List[str]:
    steps: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        step = stripped[2:].strip()
        if step:
            steps.append(step)
    return steps


def _coerce_value(token: str) -> Any:
    """Turn a string token into bool/int/float/str where sensible."""
    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    # int
    try:
        as_int = int(token)
        return as_int
    except ValueError:
        pass

    # float
    try:
        as_float = float(token)
        return as_float
    except ValueError:
        pass

    return token


def _parse_tool_line(line: str, *, allowed_tools: Set[str]) -> Optional[Dict[str, Any]]:
    stripped = line.lstrip()
    if not stripped.startswith("- "):
        return None
    body = stripped[2:].strip()
    if not body:
        return None

    parts = shlex.split(body)
    if not parts:
        return None

    name = parts[0]
    if name not in allowed_tools:
        raise PlannerParseError(f"Unknown tool name: {name!r}")

    args: Dict[str, Any] = {}

    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value_str = token.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value_str = value_str.strip()
        value = _coerce_value(value_str)
        args[key] = value

    return {"name": name, "arguments": args}


def _parse_tool_calls(lines: List[str]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for line in lines:
        parsed = _parse_tool_line(line, allowed_tools=ALL_TOOLS)
        if parsed is not None:
            tool_calls.append(parsed)
    return tool_calls


def _parse_needs_risk(lines: List[str], mode: str) -> bool:
    for line in lines:
        val = line.strip().lower()
        if not val:
            continue
        if val in {"true", "false"}:
            return val == "true"
        raise PlannerParseError(f"Invalid NEEDS_RISK_CHECK value: {val!r}")
    # default behavior: world_* modes require risk check
    return mode in {"world_query", "world_control"}


def _parse_confidence(lines: List[str]) -> float:
    for line in lines:
        val = line.strip()
        if not val:
            continue
        try:
            c = float(val)
            # don't clamp too hard; planner runner may handle this
            return c
        except ValueError:
            raise PlannerParseError(f"Invalid CONFIDENCE value: {val!r}")
    return 0.0


def parse_planner_output_v3(raw: str) -> Dict[str, Any]:
    """
    Parse the section-based V3 planner output into a payload dict compatible with
    PlannerRunner._build_plan.

    Returns:
        {
            "mode": str,
            "analysis": str,
            "steps": List[str],
            "tool_calls": List[Dict[str, Any]],
            "response_type": "text",
            "needs_risk_check": bool,
            "confidence": float,
        }

    Raises:
        PlannerParseError on structural or semantic problems.
    """
    sections = _split_sections(raw)

    mode = _parse_mode(sections["MODE"])
    analysis = _parse_analysis(sections["ANALYSIS"])
    steps = _parse_steps(sections["STEPS"])
    tool_calls = _parse_tool_calls(sections["TOOL_CALLS"])
    needs_risk_check = _parse_needs_risk(sections["NEEDS_RISK_CHECK"], mode)
    confidence = _parse_confidence(sections["CONFIDENCE"])

    return {
        "mode": mode,
        "analysis": analysis,
        "steps": steps,
        "tool_calls": tool_calls,
        "response_type": "text",  # V3 always responds with text to the user
        "needs_risk_check": needs_risk_check,
        "confidence": confidence,
    }