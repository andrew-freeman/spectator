from __future__ import annotations

from pathlib import Path

from spectator.core.tracing import TraceWriter
from spectator.core.types import ChatMessage
from spectator.runtime import checkpoints
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_default_registry


def run_turn(
    session_id: str, user_text: str, backend, base_dir: Path | None = None
) -> str:
    data_root = base_dir or checkpoints.DEFAULT_DIR.parent
    checkpoint_dir = data_root / "checkpoints"
    checkpoint = checkpoints.load_or_create(session_id, base_dir=checkpoint_dir)
    checkpoint.recent_messages.append(ChatMessage(role="user", content=user_text))

    sandbox_root = data_root / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    _registry, executor = build_default_registry(sandbox_root)

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect on the request."),
        RoleSpec(name="planner", system_prompt="Plan a response."),
        RoleSpec(name="critic", system_prompt="Critique the plan."),
        RoleSpec(name="governor", system_prompt="Decide on the final response."),
    ]

    tracer = TraceWriter(session_id, base_dir=data_root / "traces")
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
    checkpoints.save_checkpoint(updated_checkpoint, base_dir=checkpoint_dir)
    return final_text
