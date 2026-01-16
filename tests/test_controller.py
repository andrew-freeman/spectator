from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.runtime import checkpoints, controller


def test_run_turn_smoke(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.extend_role_responses(
        "reflection",
        [
            "Hello!\n"
            "<<<NOTES_JSON>>>\n"
            "{\"set_goals\":[\"ship\"],\"add_open_loops\":[\"loop\"],"
            "\"set_episode_summary\":\"summary\"}\n"
            "<<<END_NOTES_JSON>>>\n"
        ],
    )
    backend.extend_role_responses("planner", ["plan"])
    backend.extend_role_responses("critic", ["critique"])
    backend.extend_role_responses("governor", ["final answer"])

    reply = controller.run_turn("session-4", "hi", backend, base_dir=tmp_path)

    assert reply == "final answer"
    checkpoint = checkpoints.load_latest("session-4", base_dir=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert checkpoint.state.goals == ["ship"]
    assert checkpoint.state.open_loops == ["loop"]
    assert checkpoint.state.episode_summary == "summary"
    assert len(checkpoint.recent_messages) == 2
    assert (tmp_path / "traces" / "session-4.jsonl").exists()
