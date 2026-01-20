from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import time

from spectator.backends import get_backend
from spectator.core.tracing import TraceEvent, TraceWriter
from spectator.core.types import Checkpoint, State
from spectator.prompts import get_role_prompt
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools import build_readonly_registry
from spectator.analysis.chunking import chunk_file

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


def read_repo_file(
    repo_root: Path,
    path: str,
) -> str:
    target = _resolve_path(repo_root, path)
    if not target.is_file():
        raise ValueError("path is not a file")
    data = target.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES]
    return data.decode("utf-8", errors="replace")


def summarize_repo_file(
    repo_root: Path,
    path: str,
    *,
    data_root: Path,
    backend_name: str,
    max_lines: int = DEFAULT_TAIL_LINES,
    max_tokens: int | None = None,
    instruction: str | None = None,
    chunking: str = "auto",
    max_chars: int = 40000,
) -> dict[str, Any]:
    file_text = read_repo_file(repo_root, path)
    chunks = chunk_file(path, file_text, strategy=chunking, max_chars=max_chars)
    extra_instruction = instruction or "Summarize the file contents."
    backend = get_backend(backend_name)
    _registry, executor = build_readonly_registry(repo_root)

    params: dict[str, Any] = {}
    if isinstance(max_tokens, int) and max_tokens > 0:
        params["max_tokens"] = max_tokens
    roles = [
        RoleSpec(
            name="governor",
            system_prompt=get_role_prompt("governor"),
            params=params,
        ),
    ]
    checkpoint = Checkpoint(
        session_id="introspect",
        revision=0,
        updated_ts=0.0,
        state=State(),
    )
    tracer = TraceWriter("introspect", base_dir=data_root / "traces")
    map_calls = 0
    reduce_calls = 0
    total_chars = 0
    for chunk in chunks:
        total_chars += len(chunk.text)
        tracer.write(
            TraceEvent(
                ts=time.time(),
                kind="introspect_chunk",
                data={
                    "id": chunk.id,
                    "title": chunk.title,
                    "strategy": chunk.strategy,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chars": len(chunk.text),
                },
            )
        )
    footer_strategy = _resolve_chunking_strategy(path, chunking)
    if footer_strategy == "log":
        log_chunks = [chunk for chunk in chunks if _is_log_chunk(chunk)]
        nonlog_chunks = [chunk for chunk in chunks if not _is_log_chunk(chunk)]
        log_summary, log_map, log_reduce = _summarize_chunk_group(
            path,
            log_chunks,
            instruction="Summarize log events and initialization details.",
            checkpoint=checkpoint,
            roles=roles,
            backend=backend,
            executor=executor,
            tracer=tracer,
            max_chars=max_chars,
        )
        nonlog_summary, nonlog_map, nonlog_reduce = _summarize_chunk_group(
            path,
            nonlog_chunks,
            instruction="Summarize the non-log tail content.",
            checkpoint=checkpoint,
            roles=roles,
            backend=backend,
            executor=executor,
            tracer=tracer,
            max_chars=max_chars,
        )
        map_calls = log_map + nonlog_map
        reduce_calls = log_reduce + nonlog_reduce
        nonlog_lines = sum(
            chunk.end_line - chunk.start_line + 1 for chunk in nonlog_chunks
        )
        final_text = (
            f"**Log Summary**\n{log_summary}\n\n"
            f"**Non-log Tail** ({nonlog_lines} lines)\n{nonlog_summary}"
        )
    else:
        summary_text, map_calls, reduce_calls = _summarize_chunk_group(
            path,
            chunks,
            instruction=extra_instruction,
            checkpoint=checkpoint,
            roles=roles,
            backend=backend,
            executor=executor,
            tracer=tracer,
            max_chars=max_chars,
        )
        final_text = summary_text
    final_text = (
        f"{final_text}\n\nChunks: {len(chunks)} "
        f"(strategy={footer_strategy}, max_chars={max_chars})"
    )
    tracer.write(
        TraceEvent(
            ts=time.time(),
            kind="introspect_done",
            data={
                "chunks": len(chunks),
                "total_chars": total_chars,
                "map_calls": map_calls,
                "reduce_calls": reduce_calls,
            },
        )
    )
    return {
        "summary": final_text,
        "trace_file": tracer.path.name,
        "tail_lines": max_lines,
        "max_tokens": max_tokens,
        "path": path,
        "chunks": len(chunks),
        "chunking": footer_strategy,
        "max_chars": max_chars,
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


def _build_chunk_prompt(path: str, chunk, instruction: str) -> str:
    return (
        "You are in introspection mode. You may use tools to read files under the repo root.\n"
        "Available tools: fs.read_text, fs.list_dir, system.time.\n"
        f"File: {path}\n"
        f"Chunk: {chunk.title}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        f"Content:\n{chunk.text}\n\n"
        f"Task: {instruction}"
    )


def _build_reduce_prompt(
    path: str,
    chunks,
    summaries: list[str],
    instruction: str,
    max_chars: int,
) -> str:
    lines: list[str] = []
    for idx, (chunk, summary) in enumerate(zip(chunks, summaries), start=1):
        lines.append(
            f"Chunk {idx} ({chunk.title}, lines {chunk.start_line}-{chunk.end_line}):\n{summary}"
        )
    summary_block = "\n\n".join(lines)
    prefix = (
        "You are in introspection mode. You may use tools to read files under the repo root.\n"
        "Available tools: fs.read_text, fs.list_dir, system.time.\n"
        f"File: {path}\n"
        "Chunk summaries:\n"
    )
    suffix = f"\n\nTask: {instruction}"
    allowed = max_chars - len(prefix) - len(suffix)
    if allowed < 0:
        allowed = 0
    summary_block = _truncate_text(summary_block, allowed)
    prompt = f"{prefix}{summary_block}{suffix}"
    if len(prompt) > max_chars:
        prompt = _truncate_text(prompt, max_chars)
    return prompt


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = len(text) - max_chars
    marker = f"\n... <truncated {truncated} chars>"
    if len(marker) >= max_chars:
        return marker[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _run_introspect_prompt(
    prompt: str,
    *,
    checkpoint: Checkpoint,
    roles: list[RoleSpec],
    backend,
    executor,
    tracer,
) -> str:
    fresh_checkpoint = Checkpoint(
        session_id=checkpoint.session_id,
        revision=checkpoint.revision,
        updated_ts=checkpoint.updated_ts,
        state=State(),
    )
    final_text, _results, _checkpoint = run_pipeline(
        fresh_checkpoint,
        prompt,
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )
    return final_text


def _resolve_chunking_strategy(path: str, strategy: str) -> str:
    lowered = (strategy or "auto").lower()
    if lowered != "auto":
        return lowered
    suffix = Path(path).suffix.lower()
    if suffix in {".log", ".jsonl", ".txt"}:
        return "log"
    if suffix in {".md", ".rst"}:
        return "headings"
    if suffix == ".py":
        return "python_ast"
    return "fixed"


def _summarize_chunk_group(
    path: str,
    chunks: list,
    *,
    instruction: str,
    checkpoint: Checkpoint,
    roles: list[RoleSpec],
    backend,
    executor,
    tracer,
    max_chars: int,
) -> tuple[str, int, int]:
    if not chunks:
        return "No content to summarize.", 0, 0
    summaries: list[str] = []
    map_calls = 0
    for chunk in chunks:
        prompt = _build_chunk_prompt(path, chunk, instruction)
        summary = _run_introspect_prompt(
            prompt,
            checkpoint=checkpoint,
            roles=roles,
            backend=backend,
            executor=executor,
            tracer=tracer,
        )
        summaries.append(summary)
        map_calls += 1
    reduce_prompt = _build_reduce_prompt(
        path,
        chunks,
        summaries,
        instruction,
        max_chars,
    )
    final_text = _run_introspect_prompt(
        reduce_prompt,
        checkpoint=checkpoint,
        roles=roles,
        backend=backend,
        executor=executor,
        tracer=tracer,
    )
    return final_text, map_calls, 1


def _is_log_chunk(chunk) -> bool:
    return chunk.title.startswith("log ")
