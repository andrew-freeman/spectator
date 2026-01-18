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
from spectator.runtime import checkpoints, controller


class RunTurnRequest(BaseModel):
    session_id: str
    text: str
    backend: str | None = None


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
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI()
    app.state.data_root = root
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

    return app
