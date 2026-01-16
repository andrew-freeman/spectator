from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.runtime import checkpoints, controller


def test_run_turn_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checkpoints, "DEFAULT_DIR", tmp_path / "checkpoints")

    class LocalTraceWriter(controller.TraceWriter):
        def __init__(self, session_id: str) -> None:
            super().__init__(session_id, base_dir=tmp_path / "traces")

    monkeypatch.setattr(controller, "TraceWriter", LocalTraceWriter)

    backend = FakeBackend(
        responses=[
            "Hello!\n"
            "<<<NOTES_JSON>>>\n"
            "{\"set_goals\":[\"ship\"],\"add_open_loops\":[\"loop\"],"
            "\"set_episode_summary\":\"summary\"}\n"
            "<<<END_NOTES_JSON>>>\n"
        ]
    )

    reply = controller.run_turn("session-4", "hi", backend)

    assert "NOTES_JSON" not in reply
    checkpoint = checkpoints.load_latest("session-4", base_dir=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert checkpoint.state.goals == ["ship"]
    assert checkpoint.state.open_loops == ["loop"]
    assert checkpoint.state.episode_summary == "summary"
    assert len(checkpoint.recent_messages) == 2
    assert (tmp_path / "traces" / "session-4.jsonl").exists()
