from spectator.core.types import State
from spectator.runtime.condense import (
    CondensePolicy,
    condense_state,
    condense_upstream,
    truncate_text,
)
from spectator.runtime.pipeline import RoleResult


def test_condense_state_dedupe_and_cap_tail() -> None:
    state = State(
        goals=["g1", "g2", "g1", "g3"],
        open_loops=["l1", "l2", "l2", "l3"],
        decisions=["d1", "d2", "d1", "d3"],
        constraints=["c1", "c2", "c1", "c3"],
        memory_tags=["m1", "m2", "m1", "m3"],
    )
    policy = CondensePolicy(
        max_goals=2,
        max_open_loops=2,
        max_decisions=2,
        max_constraints=2,
        max_memory_tags=2,
    )

    report = condense_state(state, policy)

    assert state.goals == ["g2", "g3"]
    assert state.open_loops == ["l2", "l3"]
    assert state.decisions == ["d2", "d3"]
    assert state.constraints == ["c2", "c3"]
    assert state.memory_tags == ["m2", "m3"]
    assert report.goals_removed == 2
    assert report.open_loops_removed == 2
    assert report.decisions_removed == 2
    assert report.constraints_removed == 2
    assert report.memory_tags_removed == 2


def test_condense_upstream_truncates_with_marker() -> None:
    policy = CondensePolicy(max_upstream_chars_per_role=18, max_upstream_total_chars=25)
    results = [
        RoleResult(role="r1", text="abcdefghijklmnopqrstuvwxyz", notes=None),
        RoleResult(role="r2", text="abcdefghijklmnopqrstuvwxyz", notes=None),
    ]

    condensed = condense_upstream(results, policy)

    assert condensed[0].text == "abcd...[truncated]"
    assert condensed[1].text == truncate_text("abcdefghijklmnopqrstuvwxyz", 7)


def test_condense_noop_below_caps() -> None:
    state = State(goals=["g1"], open_loops=["l1"], decisions=["d1"], constraints=["c1"])
    policy = CondensePolicy(
        max_goals=5,
        max_open_loops=5,
        max_decisions=5,
        max_constraints=5,
        max_memory_tags=5,
        max_upstream_chars_per_role=50,
        max_upstream_total_chars=100,
    )
    results = [RoleResult(role="r1", text="short", notes=None)]

    report = condense_state(state, policy)
    condensed = condense_upstream(results, policy)

    assert state.goals == ["g1"]
    assert report.trimmed is False
    assert condensed[0].text == "short"
