from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from spectator.core.types import State
from spectator.runtime.condense import CondensePolicy, CondenseReport


class RoleResultLike(Protocol):
    text: str


@dataclass(slots=True)
class MemoryPressure:
    goals_ratio: float
    open_loops_ratio: float
    decisions_ratio: float
    constraints_ratio: float
    memory_tags_ratio: float
    upstream_ratio: float
    high_fields: list[str]
    condensed: bool
    last_report: dict[str, int] | None


def _ratio(current: int, maximum: int) -> float:
    if maximum <= 0:
        return 1.0 if current > 0 else 0.0
    return current / maximum


def compute_memory_pressure(
    state: State,
    policy: CondensePolicy,
    upstream: list[RoleResultLike],
    report: CondenseReport | None,
) -> MemoryPressure:
    goals_ratio = _ratio(len(state.goals), policy.max_goals)
    open_loops_ratio = _ratio(len(state.open_loops), policy.max_open_loops)
    decisions_ratio = _ratio(len(state.decisions), policy.max_decisions)
    constraints_ratio = _ratio(len(state.constraints), policy.max_constraints)
    memory_tags_ratio = _ratio(len(state.memory_tags), policy.max_memory_tags)
    upstream_chars = sum(len(result.text) for result in upstream)
    upstream_ratio = _ratio(upstream_chars, policy.max_upstream_total_chars)
    fields = {
        "goals_ratio": goals_ratio,
        "open_loops_ratio": open_loops_ratio,
        "decisions_ratio": decisions_ratio,
        "constraints_ratio": constraints_ratio,
        "memory_tags_ratio": memory_tags_ratio,
        "upstream_ratio": upstream_ratio,
    }
    high_fields = [name for name, ratio in fields.items() if ratio >= 0.8]
    last_report = asdict(report) if report is not None else None
    condensed = report.trimmed if report is not None else False
    return MemoryPressure(
        goals_ratio=goals_ratio,
        open_loops_ratio=open_loops_ratio,
        decisions_ratio=decisions_ratio,
        constraints_ratio=constraints_ratio,
        memory_tags_ratio=memory_tags_ratio,
        upstream_ratio=upstream_ratio,
        high_fields=high_fields,
        condensed=condensed,
        last_report=last_report,
    )


def format_memory_feedback(pressure: MemoryPressure) -> str:
    last_report = pressure.last_report if pressure.last_report is not None else "none"
    lines = [
        "=== MEMORY FEEDBACK ===",
        f"goals_ratio: {pressure.goals_ratio:.2f}",
        f"open_loops_ratio: {pressure.open_loops_ratio:.2f}",
        f"decisions_ratio: {pressure.decisions_ratio:.2f}",
        f"constraints_ratio: {pressure.constraints_ratio:.2f}",
        f"memory_tags_ratio: {pressure.memory_tags_ratio:.2f}",
        f"upstream_ratio: {pressure.upstream_ratio:.2f}",
        f"high_fields: {pressure.high_fields}",
        f"condensed: {str(pressure.condensed).lower()}",
        f"last_report: {last_report}",
        "=== END MEMORY FEEDBACK ===",
    ]
    return "\n".join(lines)
