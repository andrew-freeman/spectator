from dataclasses import dataclass

from spectator.core.types import State
from spectator.runtime.condense import CondensePolicy, CondenseReport
from spectator.runtime.memory_feedback import compute_memory_pressure


@dataclass
class DummyResult:
    text: str


def test_compute_memory_pressure_ratios_and_high_fields() -> None:
    state = State(
        goals=["g1", "g2", "g3", "g4"],
        open_loops=["l1", "l2"],
        decisions=["d1"],
        constraints=["c1", "c2", "c3"],
        memory_tags=["m1", "m2", "m3"],
    )
    policy = CondensePolicy(
        max_goals=5,
        max_open_loops=10,
        max_decisions=0,
        max_constraints=4,
        max_memory_tags=3,
        max_upstream_total_chars=10,
    )
    upstream = [DummyResult(text="12345678")]
    report = CondenseReport(goals_removed=1)

    pressure = compute_memory_pressure(state, policy, upstream, report)

    assert pressure.goals_ratio == 0.8
    assert pressure.open_loops_ratio == 0.2
    assert pressure.decisions_ratio == 1.0
    assert pressure.constraints_ratio == 0.75
    assert pressure.memory_tags_ratio == 1.0
    assert pressure.upstream_ratio == 0.8
    assert set(pressure.high_fields) == {
        "goals_ratio",
        "decisions_ratio",
        "memory_tags_ratio",
        "upstream_ratio",
    }
