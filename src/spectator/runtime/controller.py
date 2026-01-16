from __future__ import annotations

from spectator.core.tracing import TraceWriter
from spectator.core.types import ChatMessage
from spectator.runtime import checkpoints
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_default_registry


def run_turn(session_id: str, user_text: str, backend) -> str:
    checkpoint = checkpoints.load_or_create(session_id)
    checkpoint.recent_messages.append(ChatMessage(role="user", content=user_text))

    base_dir = checkpoints.DEFAULT_DIR.parent
    sandbox_root = base_dir / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    _registry, executor = build_default_registry(sandbox_root)

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect on the request."),
        RoleSpec(name="planner", system_prompt="Plan a response."),
        RoleSpec(name="critic", system_prompt="Critique the plan."),
        RoleSpec(name="governor", system_prompt="Decide on the final response."),
    ]

    tracer = TraceWriter(session_id, base_dir=base_dir / "traces")
    final_text, _results, updated_checkpoint = run_pipeline(
        checkpoint,
        user_text,
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    updated_checkpoint.recent_messages.append(
        ChatMessage(role="assistant", content=final_text)
    )
    if tracer.path.exists():
        updated_checkpoint.trace_tail = [tracer.path.name]
    checkpoints.save_checkpoint(updated_checkpoint)
    return final_text
