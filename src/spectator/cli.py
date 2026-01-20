from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from spectator.backends import get_backend, list_backends
from spectator.backends.fake import FakeBackend
from spectator.analysis.autopsy import autopsy_from_trace, render_autopsy_markdown
from spectator.analysis.introspection import (
    list_repo_files,
    read_repo_file_tail,
    resolve_repo_root,
    summarize_repo_file,
)
from spectator.runtime import controller
from spectator.runtime.tool_calls import END_MARKER, START_MARKER


class StdinBackend:
    def __init__(self, show_prompt: bool = True) -> None:
        self.show_prompt = show_prompt

    def complete(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        role = params.get("role", "assistant")
        if self.show_prompt:
            print(f"\n[{role}] prompt:\n{prompt}\n")
        return input(f"[{role}] response> ")


def _build_smoke_backend() -> FakeBackend:
    tool_calls = [{"id": "t1", "tool": "fs.list_dir", "args": {"path": "."}}]
    response_1 = (
        "Need to inspect the sandbox.\n"
        f"{START_MARKER}\n"
        f"{json.dumps(tool_calls)}\n"
        f"{END_MARKER}\n"
    )
    response_2 = "Smoke run complete."
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["Noted."])
    backend.extend_role_responses("planner", ["Plan drafted."])
    backend.extend_role_responses("critic", ["Looks good."])
    backend.extend_role_responses("governor", [response_1, response_2])
    return backend


def _resolve_backend_name(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.getenv("SPECTATOR_BACKEND", "fake")


def _build_backend(args: argparse.Namespace):
    backend_name = _resolve_backend_name(getattr(args, "backend", None))
    backend_kwargs: dict[str, Any] = {}
    if getattr(args, "model", None):
        backend_kwargs["model"] = args.model
    if getattr(args, "llama_url", None) and backend_name == "llama":
        backend_kwargs["base_url"] = args.llama_url
    return get_backend(backend_name, **backend_kwargs)


def _run_command(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    final_text = controller.run_turn(args.session, args.text, backend)
    print(final_text)
    return 0


def _repl_command(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    session_id = args.session
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip() == "/exit":
            break
        if not line.strip():
            continue
        final_text = controller.run_turn(session_id, line, backend)
        print(final_text)
    return 0


def _smoke_command(args: argparse.Namespace) -> int:
    session_id = args.session
    base_dir = Path("data") / "smoke"
    sandbox_root = base_dir / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    (sandbox_root / "hello.txt").write_text("hello", encoding="utf-8")

    backend = _build_smoke_backend()
    final_text = controller.run_turn(session_id, "Hello", backend, base_dir=base_dir)

    checkpoint_path = base_dir / "checkpoints" / f"{session_id}.json"
    trace_path = base_dir / "traces" / f"{session_id}.jsonl"

    print("Smoke run complete.")
    print(f"Final answer: {final_text}")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Trace file: {trace_path}")
    return 0


def _resolve_data_root() -> Path:
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        return Path(env_root)
    return Path("data")


def _autopsy_command(args: argparse.Namespace) -> int:
    data_root = _resolve_data_root()
    trace_path: Path | None = None
    checkpoint_path: Path | None = None
    if args.trace:
        trace_path = Path(args.trace)
    else:
        traces_dir = data_root / "traces"
        if args.session and args.run:
            trace_path = traces_dir / f"{args.session}__{args.run}.jsonl"
            checkpoint_path = data_root / "checkpoints" / f"{args.session}.json"
        else:
            if args.session:
                candidates = list(traces_dir.glob(f"{args.session}__*.jsonl"))
            else:
                candidates = list(traces_dir.glob("*.jsonl"))
            if not candidates:
                raise SystemExit("autopsy could not find any trace files")
            trace_path = max(candidates, key=lambda path: path.stat().st_mtime)
            name = trace_path.name
            if "__" in name:
                session_id = name.split("__", 1)[0]
                checkpoint_path = data_root / "checkpoints" / f"{session_id}.json"

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)

    print(f"Autopsy trace: {trace_path}")
    report = autopsy_from_trace(trace_path, checkpoint_path=checkpoint_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_autopsy_markdown(report))
    return 0


def _introspect_command(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root()
    data_root = _resolve_data_root()
    if args.list:
        files = list_repo_files(repo_root, prefix=args.path, limit=args.limit)
        print("\n".join(files))
        return 0
    if args.read:
        if not args.path:
            raise SystemExit("--path required for read")
        content = read_repo_file_tail(repo_root, args.path, max_lines=args.lines)
        print(content)
        return 0
    if args.summarize:
        if not args.path:
            raise SystemExit("--path required for summarize")
        result = summarize_repo_file(
            repo_root,
            args.path,
            data_root=data_root,
            backend_name=args.backend,
            max_lines=args.lines,
            max_tokens=args.max_tokens,
            instruction=args.instruction,
            chunking=args.chunking,
            max_chars=args.max_chars,
        )
        print(result["summary"])
        return 0
    raise SystemExit("introspect requires --list, --read, or --summarize")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single turn")
    run_parser.add_argument("--session", default="demo-1")
    run_parser.add_argument("--text", required=True)
    run_parser.add_argument("--backend", choices=list_backends())
    run_parser.add_argument("--model")
    run_parser.add_argument("--llama-url")
    run_parser.set_defaults(func=_run_command)

    repl_parser = subparsers.add_parser("repl", help="Run an interactive REPL")
    repl_parser.add_argument("--session", default="demo-1")
    repl_parser.add_argument("--backend", choices=list_backends())
    repl_parser.add_argument("--model")
    repl_parser.add_argument("--llama-url")
    repl_parser.set_defaults(func=_repl_command)

    smoke_parser = subparsers.add_parser("smoke", help="Run the smoke demo")
    smoke_parser.add_argument("--session", default="smoke-1")
    smoke_parser.set_defaults(func=_smoke_command)

    autopsy_parser = subparsers.add_parser("autopsy", help="Analyze a trace JSONL file")
    autopsy_parser.add_argument("--session")
    autopsy_parser.add_argument("--run")
    autopsy_parser.add_argument("--trace")
    autopsy_parser.add_argument("--checkpoint")
    autopsy_parser.add_argument("--json", action="store_true")
    autopsy_parser.set_defaults(func=_autopsy_command)

    introspect_parser = subparsers.add_parser("introspect", help="Read or summarize repo files")
    mode = introspect_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--read", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    introspect_parser.add_argument("--path")
    introspect_parser.add_argument("--limit", type=int, default=200)
    introspect_parser.add_argument("--lines", type=int, default=200)
    introspect_parser.add_argument("--backend", default="fake")
    introspect_parser.add_argument("--instruction")
    introspect_parser.add_argument("--max-tokens", type=int, default=1024)
    introspect_parser.add_argument(
        "--chunking",
        choices=["auto", "headings", "python_ast", "fixed", "log"],
        default="auto",
    )
    introspect_parser.add_argument("--max-chars", type=int, default=40000)
    introspect_parser.set_defaults(func=_introspect_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
