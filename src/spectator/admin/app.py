from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from spectator.admin.trace_parser import parse_trace_file
from spectator.analysis.introspection import (
    list_repo_files,
    read_repo_file_tail,
    resolve_repo_root,
    summarize_repo_file,
)
from spectator.runtime.open_loops_admin import (
    add_open_loop,
    close_open_loop,
    list_open_loops,
)
from spectator.runtime import checkpoints, controller


class RunTurnRequest(BaseModel):
    session_id: str
    text: str
    backend: str | None = None


class OpenLoopRequest(BaseModel):
    title: str
    details: str | None = None
    tags: list[str] | None = None
    priority: int | None = None


class RunOpenLoopsRequest(BaseModel):
    backend: str | None = None


class IntrospectSummarizeRequest(BaseModel):
    path: str
    backend: str | None = None
    instruction: str | None = None
    lines: int | None = None
    max_tokens: int | None = None


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        return Path(env_root)
    return checkpoints.DEFAULT_DIR.parent


def _load_checkpoint_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return {
        "session_id": payload.get("session_id"),
        "revision": payload.get("revision"),
        "updated_ts": payload.get("updated_ts"),
        "trace_tail": payload.get("trace_tail", []),
    }


def _extract_run_id(session_id: str, filename: str) -> str | None:
    prefix = f"{session_id}__"
    if not filename.startswith(prefix) or not filename.endswith(".jsonl"):
        return None
    return filename[len(prefix) : -len(".jsonl")]


def create_app(data_root: Path | None = None) -> FastAPI:
    root = _resolve_data_root(data_root)
    repo_root = resolve_repo_root()
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI()
    app.state.data_root = root
    app.state.repo_root = repo_root
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=str(template_dir))

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        token = os.getenv("SPECTATOR_ADMIN_TOKEN")
        if token:
            header = request.headers.get("X-Admin-Token")
            if header != token:
                raise HTTPException(status_code=401, detail="Invalid admin token")
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        checkpoint_dir = root / "checkpoints"
        sessions: list[dict[str, Any]] = []
        for path in checkpoint_dir.glob("*.json"):
            summary = _load_checkpoint_summary(path)
            if summary is None:
                continue
            sessions.append(summary)
        sessions.sort(key=lambda item: item.get("updated_ts") or 0, reverse=True)
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}/runs")
    async def list_runs(session_id: str) -> dict[str, Any]:
        checkpoint_path = root / "checkpoints" / f"{session_id}.json"
        trace_names: set[str] = set()
        summary = _load_checkpoint_summary(checkpoint_path)
        if summary and isinstance(summary.get("trace_tail"), list):
            trace_names.update(
                name for name in summary["trace_tail"] if isinstance(name, str)
            )
        traces_dir = root / "traces"
        for path in traces_dir.glob(f"{session_id}__*.jsonl"):
            trace_names.add(path.name)
        runs: list[dict[str, Any]] = []
        for name in sorted(trace_names):
            run_id = _extract_run_id(session_id, name)
            if run_id is None:
                continue
            runs.append({"run_id": run_id, "file_name": name})
        return {"session_id": session_id, "runs": runs}

    @app.get("/api/sessions/{session_id}/runs/{run_id}")
    async def get_run(session_id: str, run_id: str) -> dict[str, Any]:
        trace_path = root / "traces" / f"{session_id}__{run_id}.jsonl"
        if not trace_path.exists():
            raise HTTPException(status_code=404, detail="Trace file not found")
        parsed = parse_trace_file(trace_path)
        return {
            "session_id": session_id,
            "run_id": run_id,
            "file_name": trace_path.name,
            **parsed,
        }

    @app.post("/api/run_turn")
    async def run_turn(payload: RunTurnRequest) -> dict[str, Any]:
        final_text = controller.run_turn(
            payload.session_id,
            payload.text,
            base_dir=root,
            backend_name=payload.backend,
        )
        checkpoint = checkpoints.load_latest(payload.session_id, base_dir=root / "checkpoints")
        run_id = None
        trace_file_name = None
        if checkpoint is not None:
            if checkpoint.trace_tail:
                trace_file_name = checkpoint.trace_tail[-1]
                run_id = _extract_run_id(payload.session_id, trace_file_name)
            if run_id is None:
                run_id = f"rev-{checkpoint.revision}"
        if trace_file_name is None and run_id is not None:
            trace_file_name = f"{payload.session_id}__{run_id}.jsonl"
        return {
            "final_text": final_text,
            "run_id": run_id,
            "trace_file_name": trace_file_name,
            "inspect_url": f"/?session={payload.session_id}&run={run_id}",
        }

    @app.get("/api/sessions/{session_id}/open_loops")
    async def get_open_loops(session_id: str) -> dict[str, Any]:
        try:
            loops = list_open_loops(session_id, root)
        except ValueError as exc:
            if str(exc) == "session not found":
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"session_id": session_id, "open_loops": loops}

    @app.post("/api/sessions/{session_id}/open_loops")
    async def create_open_loop(session_id: str, payload: OpenLoopRequest) -> dict[str, Any]:
        try:
            loops = add_open_loop(
                session_id,
                payload.title,
                payload.details,
                payload.tags,
                payload.priority,
                root,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"session_id": session_id, "open_loops": loops}

    @app.post("/api/sessions/{session_id}/open_loops/{loop_id}/close")
    async def close_loop(session_id: str, loop_id: str) -> dict[str, Any]:
        try:
            loops = close_open_loop(session_id, loop_id, root)
        except ValueError as exc:
            if str(exc) == "session not found":
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"session_id": session_id, "open_loops": loops}

    @app.post("/api/sessions/{session_id}/open_loops/run")
    async def run_open_loops(
        session_id: str, payload: RunOpenLoopsRequest
    ) -> dict[str, Any]:
        try:
            loops = list_open_loops(session_id, root)
        except ValueError as exc:
            if str(exc) == "session not found":
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not loops:
            raise HTTPException(status_code=400, detail="no open loops to run")

        lines = [
            "Please resolve the following open loops. For each item, do the work and then close it.",
            "Emit a NOTES_JSON block with close_open_loops as a list of loop ids, using the exact markers:",
            "<<<NOTES_JSON>>>",
            "{\"close_open_loops\":[\"loop-id\"]}",
            "<<<END_NOTES_JSON>>>",
            "",
        ]
        for loop in loops:
            loop_id = loop.get("id") or "unknown"
            title = loop.get("title") or loop.get("raw") or "untitled"
            lines.append(f"- [{loop_id}] {title}")
        prompt = "\n".join(lines)

        final_text = controller.run_turn(
            session_id,
            prompt,
            base_dir=root,
            backend_name=payload.backend,
        )
        checkpoint = checkpoints.load_latest(session_id, base_dir=root / "checkpoints")
        run_id = None
        trace_file_name = None
        if checkpoint is not None and checkpoint.trace_tail:
            trace_file_name = checkpoint.trace_tail[-1]
            run_id = _extract_run_id(session_id, trace_file_name)
        if run_id is None and checkpoint is not None:
            run_id = f"rev-{checkpoint.revision}"
        if trace_file_name is None and run_id is not None:
            trace_file_name = f"{session_id}__{run_id}.jsonl"

        updated_loops = list_open_loops(session_id, root)
        return {
            "session_id": session_id,
            "run_id": run_id,
            "trace_file_name": trace_file_name,
            "final_text": final_text,
            "open_loops": updated_loops,
        }

    @app.get("/api/introspect/list")
    async def list_introspect(path: str | None = None, limit: int = 500) -> dict[str, Any]:
        files = list_repo_files(repo_root, prefix=path, limit=limit)
        return {"root": str(repo_root), "files": files}

    @app.get("/api/introspect/read")
    async def read_introspect(path: str, lines: int = 200) -> dict[str, Any]:
        try:
            content = read_repo_file_tail(repo_root, path, max_lines=lines)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"path": path, "lines": lines, "content": content}

    @app.post("/api/introspect/summarize")
    async def summarize_introspect(
        payload: IntrospectSummarizeRequest,
    ) -> dict[str, Any]:
        path = payload.path
        backend = payload.backend or "fake"
        instruction = payload.instruction
        lines = payload.lines or 200
        max_tokens = payload.max_tokens if payload.max_tokens is not None else 1024
        if not isinstance(lines, int) or lines <= 0:
            raise HTTPException(status_code=400, detail="lines must be positive")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise HTTPException(status_code=400, detail="max_tokens must be positive")
        result = summarize_repo_file(
            repo_root,
            path,
            data_root=root,
            backend_name=backend,
            max_lines=lines,
            max_tokens=max_tokens,
            instruction=instruction if isinstance(instruction, str) else None,
        )
        return result

    return app
