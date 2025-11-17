# app/critic/critic_output_parser_v3.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


SECTION_ORDER = ["RISK_LEVEL", "CONFIDENCE", "ISSUES", "NOTES"]


class CriticParseError(ValueError):
    """Raised when the critic output cannot be parsed according to the V3 schema."""


def _split_sections(raw: str) -> Dict[str, List[str]]:
    lines = (raw or "").splitlines()
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_ORDER}
    current: Optional[str] = None

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("#RISK_LEVEL:"):
            current = "RISK_LEVEL"
            continue
        if stripped.startswith("#CONFIDENCE:"):
            current = "CONFIDENCE"
            continue
        if stripped.startswith("#ISSUES:"):
            current = "ISSUES"
            continue
        if stripped.startswith("#NOTES:"):
            current = "NOTES"
            continue

        if current is not None:
            sections[current].append(stripped)

    return sections


def _parse_risk_level(lines: List[str]) -> str:
    for line in lines:
        val = line.strip().lower()
        if not val:
            continue
        if val in {"low", "medium", "high", "unsafe"}:
            return val
        raise CriticParseError(f"Invalid RISK_LEVEL value: {val!r}")
    # Default to medium if missing (conservative but not catastrophic)
    return "medium"


def _parse_confidence(lines: List[str]) -> float:
    for line in lines:
        val = line.strip()
        if not val:
            continue
        try:
            return float(val)
        except ValueError:
            raise CriticParseError(f"Invalid CONFIDENCE value: {val!r}")
    return 0.0


def _parse_issues(lines: List[str]) -> List[str]:
    issues: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].strip()
        if item:
            issues.append(item)
    if not issues:
        return []
    # Normalize explicit "none" to empty list
    if len(issues) == 1 and issues[0].lower() == "none":
        return []
    return issues


def _parse_notes(lines: List[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return text


def parse_critic_output_v3(raw: str) -> Dict[str, Any]:
    """
    Parse the section-based V3 critic output into a payload dict compatible
    with CriticOutput.

    Returns:
        {
            "risk_level": "low" | "medium" | "high" | "unsafe",
            "confidence": float,
            "detected_issues": List[str],
            "notes": str,
        }
    """
    sections = _split_sections(raw)

    risk_level = _parse_risk_level(sections["RISK_LEVEL"])
    confidence = _parse_confidence(sections["CONFIDENCE"])
    detected_issues = _parse_issues(sections["ISSUES"])
    notes = _parse_notes(sections["NOTES"])

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "detected_issues": detected_issues,
        "notes": notes,
    }