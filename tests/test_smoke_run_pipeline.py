import json

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.runtime.checkpoints import load_or_create
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler
from spectator.tools.registry import ToolRegistry


def test_smoke_run_pipeline_executes_tool_calls(tmp_path) -> None:
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    (sandbox_root / "hello.txt").write_text("hello", encoding="utf-8")

    registry = ToolRegistry()
    registry.register("fs.list_dir", list_dir_handler(sandbox_root))
    executor = ToolExecutor(root=sandbox_root, registry=registry)

    tracer = TraceWriter("smoke-1", base_dir=tmp_path / "traces")
    checkpoint = load_or_create("smoke-1", base_dir=tmp_path / "checkpoints")

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect."),
        RoleSpec(name="planner", system_prompt="Plan."),
        RoleSpec(name="critic", system_prompt="Critique."),
        RoleSpec(name="governor", system_prompt="Decide."),
    ]

    tool_calls = [{"id": "t1", "tool": "fs.list_dir", "args": {"path": "."}}]
    response_1 = (
        "Need a tool.\n"
        "<<<TOOL_CALLS_JSON>>>\n"
        f"{json.dumps(tool_calls)}\n"
        "<<<END_TOOL_CALLS_JSON>>>\n"
    )
    response_2 = "done"

    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["reflection"])
    backend.extend_role_responses("planner", ["planner"])
    backend.extend_role_responses("critic", ["critic"])
    backend.extend_role_responses("governor", [response_1, response_2])

    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        "hello",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    assert final_text == response_2

    trace_lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in trace_lines]
    tool_done = [event for event in events if event["kind"] == "tool_done"]
    assert any(
        event["data"].get("tool") == "fs.list_dir" and event["data"].get("ok") is True
        for event in tool_done
    )
