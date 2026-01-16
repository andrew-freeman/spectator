from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

from spectator.core.types import State

TRUNCATION_MARKER = "...[truncated]"


@dataclass(slots=True)
class CondensePolicy:
    max_goals: int = 32
    max_open_loops: int = 32
    max_decisions: int = 32
    max_constraints: int = 32
    max_memory_tags: int = 32
    max_upstream_chars_per_role: int = 1500
    max_upstream_total_chars: int = 4000


@dataclass(slots=True)
class CondenseReport:
    goals_removed: int = 0
    open_loops_removed: int = 0
    decisions_removed: int = 0
    constraints_removed: int = 0
    memory_tags_removed: int = 0

    @property
    def trimmed(self) -> bool:
        return any(
            (
                self.goals_removed,
                self.open_loops_removed,
                self.decisions_removed,
                self.constraints_removed,
                self.memory_tags_removed,
            )
        )


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def cap_tail(items: Iterable[str], max_n: int) -> list[str]:
    materialized = list(items)
    if max_n < 0:
        return []
    if len(materialized) <= max_n:
        return materialized
    if max_n == 0:
        return []
    return materialized[-max_n:]


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars < len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars]
    available = max_chars - len(TRUNCATION_MARKER)
    return f"{text[:available]}{TRUNCATION_MARKER}"


def _condense_list(items: Iterable[str], max_items: int) -> list[str]:
    return cap_tail(dedupe_preserve_order(items), max_items)


def condense_state(state: State, policy: CondensePolicy) -> CondenseReport:
    before_goals = len(state.goals)
    before_open_loops = len(state.open_loops)
    before_decisions = len(state.decisions)
    before_constraints = len(state.constraints)
    before_memory_tags = len(state.memory_tags)

    state.goals = _condense_list(state.goals, policy.max_goals)
    state.open_loops = _condense_list(state.open_loops, policy.max_open_loops)
    state.decisions = _condense_list(state.decisions, policy.max_decisions)
    state.constraints = _condense_list(state.constraints, policy.max_constraints)
    state.memory_tags = _condense_list(state.memory_tags, policy.max_memory_tags)

    return CondenseReport(
        goals_removed=before_goals - len(state.goals),
        open_loops_removed=before_open_loops - len(state.open_loops),
        decisions_removed=before_decisions - len(state.decisions),
        constraints_removed=before_constraints - len(state.constraints),
        memory_tags_removed=before_memory_tags - len(state.memory_tags),
    )


RoleResultT = TypeVar("RoleResultT")


def _rebuild_role_result(result: RoleResultT, text: str) -> RoleResultT:
    return result.__class__(role=result.role, text=text, notes=result.notes)


def condense_upstream(results: list[RoleResultT], policy: CondensePolicy) -> list[RoleResultT]:
    truncated_results = [
        _rebuild_role_result(
            result,
            truncate_text(result.text, policy.max_upstream_chars_per_role),
        )
        for result in results
    ]

    total_chars = sum(len(result.text) for result in truncated_results)
    if total_chars <= policy.max_upstream_total_chars:
        return truncated_results

    condensed: list[RoleResultT] = []
    remaining = policy.max_upstream_total_chars
    for result in truncated_results:
        text = truncate_text(result.text, remaining)
        condensed.append(_rebuild_role_result(result, text))
        remaining -= len(text)
        if remaining <= 0:
            remaining = 0

    return condensed
