#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.core.types import Checkpoint, State
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler
from spectator.tools.registry import ToolRegistry
from spectator.tools.settings import default_tool_settings
from spectator.tools.shell_tool import shell_exec_handler


SESSION_ID = f"tool-exec-ci-{time.strftime('%Y%m%d-%H%M%S')}"
ROOT = Path.cwd()
ARTIFACTS_ROOT = Path(
    os.getenv("ARTIFACTS_ROOT", str(ROOT / "llama_test_artifacts"))
)
OUTDIR = ARTIFACTS_ROOT / SESSION_ID / "tool_exec_cases"


@dataclass
class Case:
    case_id: str
    prompt: str
    responses: list[str]
    tool: str
    args: dict[str, Any]
    tool_result_markers: list[str]
    expect_visible: str


def _http_stub_handler(args: dict[str, Any], _context) -> dict[str, Any]:
    # Deterministic stub: never hits the network.
    url = args.get("url")
    return {"url": url, "status": 200, "text": "stubbed http payload", "cache_hit": False}


def _build_executor(root: Path) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register("fs.list_dir", list_dir_handler(root))
    registry.register("shell.exec", shell_exec_handler(root))
    registry.register("http.get", _http_stub_handler)
    settings = default_tool_settings(root)
    return ToolExecutor(root, registry, settings)


def _load_events(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in raw if line.strip()]


def _assert_tool_events(events: list[dict[str, Any]], tool: str, args: dict[str, Any]) -> None:
    starts = [event for event in events if event.get("kind") == "tool_start"]
    dones = [event for event in events if event.get("kind") == "tool_done"]
    if not any(start.get("data", {}).get("tool") == tool for start in starts):
        raise AssertionError(f"missing tool_start for {tool}")
    if not any(done.get("data", {}).get("tool") == tool for done in dones):
        raise AssertionError(f"missing tool_done for {tool}")
    for start in starts:
        if start.get("data", {}).get("tool") == tool:
            if start.get("data", {}).get("args") != args:
                raise AssertionError("tool_start args mismatch")
            break


def _assert_no_tool_json_leak(visible_output: str) -> None:
    lowered = visible_output.lower()
    forbidden = ["fs.", "shell.", "http.", "{\"name\"", "{\"tool\""]
    for marker in forbidden:
        if marker in lowered:
            raise AssertionError(f"tool leak marker found: {marker}")


def _run_case(case: Case) -> None:
    case_dir = OUTDIR / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(case.prompt, encoding="utf-8")

    sandbox = case_dir / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "hello.txt").write_text("hello", encoding="utf-8")
    executor = _build_executor(sandbox)

    backend = FakeBackend()
    backend.set_role_responses("governor", case.responses)
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter(case.case_id, base_dir=case_dir / "traces")

    state = State()
    if case.tool == "http.get":
        state.capabilities_granted = ["net"]
    checkpoint = Checkpoint(session_id=case.case_id, revision=0, updated_ts=0.0, state=state)

    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        case.prompt,
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    if len(backend.calls) < 2:
        raise AssertionError("expected tool results round-trip")
    tool_prompt = backend.calls[1]["prompt"]
    if "TOOL_RESULTS:" not in tool_prompt:
        raise AssertionError("TOOL_RESULTS block missing from follow-up prompt")
    for marker in case.tool_result_markers:
        if marker not in tool_prompt:
            raise AssertionError(f"missing tool result marker: {marker}")

    (case_dir / "output_only.txt").write_text(final_text, encoding="utf-8", errors="replace")

    events = _load_events(tracer.path)
    _assert_tool_events(events, case.tool, case.args)

    visible_events = [event for event in events if event.get("kind") == "visible_response"]
    visible_output = visible_events[-1]["data"]["visible_response"]
    _assert_no_tool_json_leak(visible_output)

    if case.expect_visible not in final_text:
        raise AssertionError("final visible output mismatch")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = [
        Case(
            case_id="LLM_06_tool_execution_fs",
            prompt="List the sandbox contents.",
            responses=[
                "{\"name\":\"fs.list_dir\",\"arguments\":{\"path\":\".\"}}",
                "Listed sandbox entries: {{TOOL_OUTPUT}}",
            ],
            tool="fs.list_dir",
            args={"path": "."},
            tool_result_markers=["entries", "hello.txt"],
            expect_visible="hello.txt",
        ),
        Case(
            case_id="LLM_07_tool_execution_shell",
            prompt="Run a safe shell command.",
            responses=[
                "Run shell.\n"
                "<<<TOOL_CALLS_JSON>>>\n"
                "[{\"id\":\"call-1\",\"tool\":\"shell.exec\",\"args\":{\"cmd\":\"echo hello-from-shell\"}}]\n"
                "<<<END_TOOL_CALLS_JSON>>>\n",
                "Shell output captured: {{TOOL_OUTPUT}}",
            ],
            tool="shell.exec",
            args={"cmd": "echo hello-from-shell"},
            tool_result_markers=["hello-from-shell"],
            expect_visible="hello-from-shell",
        ),
        Case(
            case_id="LLM_08_tool_execution_http",
            prompt="Fetch the HTTP response.",
            responses=[
                "{\"tool\":\"http.get\",\"args\":{\"url\":\"https://example.invalid\"}}",
                "HTTP response summarized: {{TOOL_OUTPUT}}",
            ],
            tool="http.get",
            args={"url": "https://example.invalid"},
            tool_result_markers=["stubbed http payload"],
            expect_visible="stubbed http payload",
        ),
    ]

    for case in cases:
        _run_case(case)

    print(f"tool exec cases written to: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
