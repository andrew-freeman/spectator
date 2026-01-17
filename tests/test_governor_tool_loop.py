import json

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.core.types import Checkpoint, State
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_default_registry


def test_governor_tool_loop_executes_tools_and_traces(tmp_path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "hello.txt").write_text("hello", encoding="utf-8")
    _registry, executor = build_default_registry(sandbox)

    tool_calls = [{"id": "t1", "tool": "fs.list_dir", "args": {"path": "."}}]
    response_1 = (
        "Need a tool.\n"
        "<<<TOOL_CALLS_JSON>>>\n"
        f"{json.dumps(tool_calls)}\n"
        "<<<END_TOOL_CALLS_JSON>>>\n"
    )
    response_2 = "done"

    backend = FakeBackend()
    backend.extend_role_responses("governor", [response_1, response_2])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter("session-1", base_dir=tmp_path / "traces")

    checkpoint = Checkpoint(session_id="s-1", revision=0, updated_ts=0.0, state=State())

    run_pipeline(
        checkpoint,
        "hello",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    assert len(backend.calls) == 2
    assert "TOOL_RESULTS:" in backend.calls[1]["prompt"]
    assert "fs.list_dir" in backend.calls[1]["prompt"]
    assert "hello.txt" in backend.calls[1]["prompt"]

    trace_lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    kinds = {json.loads(line)["kind"] for line in trace_lines}
    assert {"tool_plan", "tool_start", "tool_done"}.issubset(kinds)


def test_governor_tool_loop_strips_reasoning_but_keeps_tool_calls(tmp_path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "hello.txt").write_text("hello", encoding="utf-8")
    _registry, executor = build_default_registry(sandbox)

    tool_calls = [{"id": "t1", "tool": "fs.list_dir", "args": {"path": "."}}]
    response_1 = (
        "Need a tool.\n"
        "<think>ignore</think>\n"
        "<<<TOOL_CALLS_JSON>>>\n"
        f"{json.dumps(tool_calls)}\n"
        "<<<END_TOOL_CALLS_JSON>>>\n"
    )
    response_2 = "<think>hidden</think>Final answer."

    backend = FakeBackend()
    backend.extend_role_responses("governor", [response_1, response_2])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]

    checkpoint = Checkpoint(session_id="s-2", revision=0, updated_ts=0.0, state=State())

    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        "hello",
        roles,
        backend,
        tool_executor=executor,
    )

    assert len(backend.calls) == 2
    assert "TOOL_RESULTS:" in backend.calls[1]["prompt"]
    assert final_text == "Final answer."
