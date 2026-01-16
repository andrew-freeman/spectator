from spectator.backends.fake import FakeBackend
from spectator.core.types import Checkpoint, State
from spectator.runtime.pipeline import RoleSpec, run_pipeline


def test_pipeline_injects_upstream_content() -> None:
    checkpoint = Checkpoint(session_id="s-1", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])
    backend.extend_role_responses("planner", ["planner output"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect."),
        RoleSpec(name="planner", system_prompt="Plan."),
    ]

    run_pipeline(checkpoint, "hello", roles, backend)

    assert "UPSTREAM" not in backend.calls[0]["prompt"]
    assert "reflection: reflection output" in backend.calls[1]["prompt"]


def test_pipeline_applies_notes_patch_and_strips_notes() -> None:
    checkpoint = Checkpoint(
        session_id="s-2",
        revision=0,
        updated_ts=0.0,
        state=State(open_loops=["loop-1"]),
    )
    backend = FakeBackend()
    backend.extend_role_responses(
        "reflection",
        [
            "Draft.\n"
            "<<<NOTES_JSON>>>\n"
            "{\"set_goals\":[\"ship\"],\"add_open_loops\":[\"loop-2\"],"
            "\"close_open_loops\":[\"loop-1\"],\"add_constraints\":[\"constraint\"],"
            "\"set_episode_summary\":\"summary\"}\n"
            "<<<END_NOTES_JSON>>>\n"
        ],
    )
    backend.extend_role_responses("governor", ["final answer"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect."),
        RoleSpec(name="governor", system_prompt="Decide."),
    ]

    final_text, results, updated = run_pipeline(checkpoint, "hello", roles, backend)

    assert "NOTES_JSON" not in results[0].text
    assert updated.state.goals == ["ship"]
    assert updated.state.open_loops == ["loop-2"]
    assert updated.state.constraints == ["constraint"]
    assert updated.state.episode_summary == "summary"
    assert final_text == "final answer"


def test_pipeline_injects_telemetry_for_enabled_roles() -> None:
    checkpoint = Checkpoint(session_id="s-3", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])
    backend.extend_role_responses("planner", ["planner output"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect.", telemetry="basic"),
        RoleSpec(name="planner", system_prompt="Plan.", telemetry="none"),
    ]

    run_pipeline(checkpoint, "hello", roles, backend)

    assert "=== TELEMETRY (basic) ===" in backend.calls[0]["prompt"]
    assert "=== TELEMETRY (basic) ===" not in backend.calls[1]["prompt"]


def test_pipeline_only_injects_telemetry_for_requested_roles() -> None:
    checkpoint = Checkpoint(session_id="s-4", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])
    backend.extend_role_responses("governor", ["final answer"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect.", telemetry="none"),
        RoleSpec(name="governor", system_prompt="Decide.", telemetry="basic"),
    ]

    run_pipeline(checkpoint, "hello", roles, backend)

    assert "=== TELEMETRY (basic) ===" not in backend.calls[0]["prompt"]
    assert "=== TELEMETRY (basic) ===" in backend.calls[1]["prompt"]
