from __future__ import annotations

import os
from pathlib import Path

from spectator.backends import get_backend
from spectator.core.tracing import TraceWriter
from spectator.core.types import ChatMessage
from spectator.runtime import checkpoints
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_default_registry


def run_turn(
    session_id: str,
    user_text: str,
    backend=None,
    base_dir: Path | None = None,
    *,
    backend_name: str | None = None,
    backend_params: dict[str, object] | None = None,
) -> str:
    if backend is None:
        name = backend_name or os.getenv("SPECTATOR_BACKEND", "fake")
        backend = get_backend(name, **(backend_params or {}))
    data_root = base_dir or checkpoints.DEFAULT_DIR.parent
    checkpoint_dir = data_root / "checkpoints"
    checkpoint = checkpoints.load_or_create(session_id, base_dir=checkpoint_dir)
    checkpoint.recent_messages.append(ChatMessage(role="user", content=user_text))

    sandbox_root = data_root / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    _registry, executor = build_default_registry(sandbox_root)

    safety_suffix = "Don't output chain-of-thought; output only final answer."
    roles = [
        RoleSpec(
            name="reflection",
            system_prompt=f"Reflect on the request. {safety_suffix}",
        ),
        RoleSpec(
            name="planner",
            system_prompt=f"Plan a response. {safety_suffix}",
        ),
        RoleSpec(
            name="critic",
            system_prompt=f"Critique the plan. {safety_suffix}",
        ),
        RoleSpec(
            name="governor",
            system_prompt=f"Decide on the final response. {safety_suffix}",
        ),
    ]

    run_id = f"rev-{checkpoint.revision + 1}"
    tracer = TraceWriter(session_id, base_dir=data_root / "traces", run_id=run_id)
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
        trace_name = tracer.path.name
        if trace_name not in updated_checkpoint.trace_tail:
            updated_checkpoint.trace_tail.append(trace_name)
        if len(updated_checkpoint.trace_tail) > 20:
            updated_checkpoint.trace_tail = updated_checkpoint.trace_tail[-20:]
    checkpoints.save_checkpoint(updated_checkpoint, base_dir=checkpoint_dir)
    return final_text
