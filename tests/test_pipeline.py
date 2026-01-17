import json

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.core.types import ChatMessage, Checkpoint, State
from spectator.memory.embeddings import HashEmbedder
from spectator.memory.vector_store import MemoryRecord, SQLiteVectorStore
from spectator.prompts import get_role_prompt
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


def test_pipeline_injects_memory_feedback_for_enabled_roles() -> None:
    checkpoint = Checkpoint(session_id="s-5", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])
    backend.extend_role_responses("planner", ["planner output"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect.", memory_feedback="basic"),
        RoleSpec(name="planner", system_prompt="Plan.", memory_feedback="none"),
    ]

    run_pipeline(checkpoint, "hello", roles, backend)

    assert "=== MEMORY FEEDBACK ===" in backend.calls[0]["prompt"]
    assert "=== MEMORY FEEDBACK ===" not in backend.calls[1]["prompt"]


def test_pipeline_memory_feedback_marks_condensed_state() -> None:
    checkpoint = Checkpoint(session_id="s-6", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    goals = ",".join([f"\"goal-{idx}\"" for idx in range(40)])
    backend.extend_role_responses(
        "reflection",
        [
            "Draft.\n"
            "<<<NOTES_JSON>>>\n"
            f"{{\"set_goals\":[{goals}]}}\n"
            "<<<END_NOTES_JSON>>>\n"
        ],
    )
    backend.extend_role_responses("governor", ["final answer"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect.", memory_feedback="none"),
        RoleSpec(name="governor", system_prompt="Decide.", memory_feedback="basic"),
    ]

    run_pipeline(checkpoint, "hello", roles, backend)

    assert "=== MEMORY FEEDBACK ===" in backend.calls[1]["prompt"]
    assert "condensed: true" in backend.calls[1]["prompt"]


def test_pipeline_injects_retrieval_block_for_enabled_roles(tmp_path) -> None:
    checkpoint = Checkpoint(session_id="s-7", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])
    backend.extend_role_responses("planner", ["planner output"])

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect.", wants_retrieval=True),
        RoleSpec(name="planner", system_prompt="Plan.", wants_retrieval=False),
    ]

    store = SQLiteVectorStore(tmp_path / "memory.sqlite")
    embedder = HashEmbedder(dim=32)
    record = MemoryRecord(id="mem-1", ts=1.0, text="remember this detail")
    store.add([record], embedder.embed([record.text]))
    memory = type("Memory", (), {"store": store, "embedder": embedder})()

    run_pipeline(checkpoint, "remember this detail", roles, backend, memory=memory)

    assert "=== RETRIEVAL ===" in backend.calls[0]["prompt"]
    assert "=== END RETRIEVAL ===" in backend.calls[0]["prompt"]
    assert "=== RETRIEVAL ===" not in backend.calls[1]["prompt"]


def test_pipeline_omits_retrieval_block_without_request(tmp_path) -> None:
    checkpoint = Checkpoint(session_id="s-8", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])

    roles = [RoleSpec(name="reflection", system_prompt="Reflect.", wants_retrieval=False)]

    store = SQLiteVectorStore(tmp_path / "memory.sqlite")
    embedder = HashEmbedder(dim=32)
    record = MemoryRecord(id="mem-2", ts=1.0, text="memory text")
    store.add([record], embedder.embed([record.text]))
    memory = type("Memory", (), {"store": store, "embedder": embedder})()

    run_pipeline(checkpoint, "memory text", roles, backend, memory=memory)

    assert "=== RETRIEVAL ===" not in backend.calls[0]["prompt"]


def test_pipeline_traces_visible_response(tmp_path) -> None:
    checkpoint = Checkpoint(session_id="s-9", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("governor", ["<think>secret</think>Visible"])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter("session-9", base_dir=tmp_path / "traces")

    final_text, _results, _updated = run_pipeline(
        checkpoint, "hello", roles, backend, tracer=tracer
    )

    trace_lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    llm_done = [line for line in trace_lines if '"kind": "llm_done"' in line][0]
    visible_response = [
        line for line in trace_lines if '"kind": "visible_response"' in line
    ][0]
    assert "<think>secret</think>Visible" in llm_done
    assert '"visible_response": "Visible"' in visible_response
    assert final_text == "Visible"


def test_pipeline_traces_streaming_deltas(tmp_path) -> None:
    class StreamingBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete(self, prompt: str, params: dict[str, object] | None = None) -> str:
            params = params or {}
            self.calls.append({"prompt": prompt, "params": params})
            deltas = ["alpha ", "beta ", "gamma "]
            callback = params.get("stream_callback")
            if params.get("stream") and callable(callback):
                for delta in deltas:
                    callback(delta)
            return "".join(deltas) + "done"

    checkpoint = Checkpoint(session_id="s-10", revision=0, updated_ts=0.0, state=State())
    backend = StreamingBackend()
    roles = [RoleSpec(name="governor", system_prompt="Decide.", params={"stream": True})]
    tracer = TraceWriter("session-10", base_dir=tmp_path / "traces")

    run_pipeline(checkpoint, "hello", roles, backend, tracer=tracer)

    events = [
        json.loads(line)
        for line in tracer.path.read_text(encoding="utf-8").strip().splitlines()
    ]
    stream_events = [event for event in events if event["kind"] == "llm_stream"]
    assert [event["data"]["delta"] for event in stream_events] == [
        "alpha ",
        "beta ",
        "gamma ",
    ]
    kinds = [event["kind"] for event in events]
    assert kinds.index("llm_req") < kinds.index("llm_stream") < kinds.index("llm_done")


def test_pipeline_includes_history_block_when_messages_present() -> None:
    checkpoint = Checkpoint(
        session_id="s-11",
        revision=0,
        updated_ts=0.0,
        state=State(),
        recent_messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ],
    )
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])

    roles = [RoleSpec(name="reflection", system_prompt=get_role_prompt("reflection"))]

    run_pipeline(checkpoint, "hello", roles, backend)

    prompt = backend.calls[0]["prompt"]
    assert "HISTORY:\nuser: Hello\nassistant: Hi there" in prompt
    assert prompt.count("HISTORY:") == 1


def test_pipeline_includes_empty_history_instruction_when_absent() -> None:
    checkpoint = Checkpoint(session_id="s-12", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection output"])

    roles = [RoleSpec(name="reflection", system_prompt=get_role_prompt("reflection"))]

    run_pipeline(checkpoint, "hello", roles, backend)

    prompt = backend.calls[0]["prompt"]
    assert "If HISTORY is empty, say so and ask for context." in prompt
    assert "HISTORY:\n(empty)" in prompt
    assert prompt.count("HISTORY:") == 1
