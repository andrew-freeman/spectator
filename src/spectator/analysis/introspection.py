from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from spectator.backends import get_backend
from spectator.core.tracing import TraceWriter
from spectator.core.types import Checkpoint, State
from spectator.prompts import get_role_prompt
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_readonly_registry

MAX_FILE_BYTES = 1_000_000
DEFAULT_TAIL_LINES = 200
DEFAULT_LIST_LIMIT = 500


def resolve_repo_root() -> Path:
    env_root = os.getenv("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


def list_repo_files(
    repo_root: Path,
    prefix: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> list[str]:
    target = _resolve_path(repo_root, prefix or ".")
    if target.is_file():
        return [str(target.relative_to(repo_root))]
    results: list[str] = []
    for path in sorted(target.rglob("*")):
        if path.is_file():
            results.append(str(path.relative_to(repo_root)))
            if len(results) >= limit:
                break
    return results


def read_repo_file_tail(
    repo_root: Path,
    path: str,
    max_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    target = _resolve_path(repo_root, path)
    if not target.is_file():
        raise ValueError("path is not a file")
    data = target.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        data = data[-MAX_FILE_BYTES:]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def summarize_repo_file(
    repo_root: Path,
    path: str,
    *,
    data_root: Path,
    backend_name: str,
    max_lines: int = DEFAULT_TAIL_LINES,
    instruction: str | None = None,
) -> dict[str, Any]:
    tail = read_repo_file_tail(repo_root, path, max_lines=max_lines)
    extra_instruction = instruction or "Summarize the file contents."
    prompt = (
        "You are in introspection mode. You may use tools to read files under the repo root.\n"
        "Available tools: fs.read_text, fs.list_dir, system.time.\n"
        f"File: {path}\n"
        f"Tail ({max_lines} lines):\n"
        f"{tail}\n\n"
        f"Task: {extra_instruction}"
    )

    backend = get_backend(backend_name)
    _registry, executor = build_readonly_registry(repo_root)

    roles = [
        RoleSpec(
            name="governor",
            system_prompt=get_role_prompt("governor"),
        ),
    ]
    checkpoint = Checkpoint(
        session_id="introspect",
        revision=0,
        updated_ts=0.0,
        state=State(),
    )
    tracer = TraceWriter("introspect", base_dir=data_root / "traces")

    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        prompt,
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )
    return {
        "summary": final_text,
        "trace_file": tracer.path.name,
        "tail_lines": max_lines,
        "path": path,
    }


def _resolve_path(repo_root: Path, path: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("path escapes repo root") from exc
    return resolved
