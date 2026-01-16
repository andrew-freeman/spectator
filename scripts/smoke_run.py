from __future__ import annotations

import json
from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.runtime.checkpoints import load_or_create, save_checkpoint
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler, read_text_handler
from spectator.tools.registry import ToolRegistry


def main() -> None:
    session_id = "smoke-1"
    base_dir = Path("data") / "smoke"
    sandbox_root = base_dir / "sandbox"
    trace_dir = base_dir / "traces"
    checkpoint_dir = base_dir / "checkpoints"

    sandbox_root.mkdir(parents=True, exist_ok=True)
    (sandbox_root / "hello.txt").write_text("hello", encoding="utf-8")

    registry = ToolRegistry()
    registry.register("fs.list_dir", list_dir_handler(sandbox_root))
    registry.register("fs.read_text", read_text_handler(sandbox_root))

    executor = ToolExecutor(root=sandbox_root, registry=registry)
    tracer = TraceWriter(session_id, base_dir=trace_dir)
    checkpoint = load_or_create(session_id, base_dir=checkpoint_dir)

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect briefly."),
        RoleSpec(name="planner", system_prompt="Plan the response."),
        RoleSpec(name="critic", system_prompt="Critique the plan."),
        RoleSpec(name="governor", system_prompt="Use tools and answer."),
    ]

    tool_calls = [{"id": "t1", "tool": "fs.list_dir", "args": {"path": "."}}]
    response_1 = (
        "Need to inspect sandbox.\n"
        "<<<TOOL_CALLS_JSON>>>\n"
        f"{json.dumps(tool_calls)}\n"
        "<<<END_TOOL_CALLS_JSON>>>\n"
    )
    response_2 = "Smoke run complete."

    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["Noted."])
    backend.extend_role_responses("planner", ["Plan drafted."])
    backend.extend_role_responses("critic", ["Looks good."])
    backend.extend_role_responses("governor", [response_1, response_2])

    final_text, _results, updated_checkpoint = run_pipeline(
        checkpoint,
        "Hello",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    checkpoint_path = save_checkpoint(updated_checkpoint, base_dir=checkpoint_dir)

    print("Final answer:")
    print(final_text)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Trace file: {tracer.path}")


if __name__ == "__main__":
    main()
