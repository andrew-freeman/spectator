from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.runtime import checkpoints, controller


def test_run_turn_smoke(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.extend_role_responses(
        "reflection",
        ["Hello!"],
    )
    backend.extend_role_responses("planner", ["plan"])
    backend.extend_role_responses("critic", ["critique"])
    backend.extend_role_responses(
        "governor",
        [
            "final answer\n"
            "<<<NOTES_JSON>>>\n"
            "{\"set_goals\":[\"ship\"],\"add_open_loops\":[\"loop\"],"
            "\"set_episode_summary\":\"summary\"}\n"
            "<<<END_NOTES_JSON>>>\n"
        ],
    )

    reply = controller.run_turn("session-4", "hi", backend, base_dir=tmp_path)

    assert reply.strip() == "final answer"
    checkpoint = checkpoints.load_latest("session-4", base_dir=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert checkpoint.state.goals == ["ship"]
    assert checkpoint.state.open_loops == ["loop"]
    assert checkpoint.state.episode_summary == "summary"
    assert len(checkpoint.recent_messages) == 2
    assert (tmp_path / "traces" / "session-4__rev-1.jsonl").exists()


def test_run_turn_appends_trace_tail_and_caps(tmp_path: Path) -> None:
    backend = FakeBackend()
    turns = 25
    backend.extend_role_responses("reflection", ["ok"] * turns)
    backend.extend_role_responses("planner", ["ok"] * turns)
    backend.extend_role_responses("critic", ["ok"] * turns)
    backend.extend_role_responses("governor", ["final"] * turns)

    for idx in range(turns):
        controller.run_turn("session-tail", f"hi {idx}", backend, base_dir=tmp_path)

    checkpoint = checkpoints.load_latest("session-tail", base_dir=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert len(checkpoint.trace_tail) == 20
    expected_tail = [
        f"session-tail__rev-{rev}.jsonl" for rev in range(turns - 19, turns + 1)
    ]
    assert checkpoint.trace_tail == expected_tail
