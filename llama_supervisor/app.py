from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llama_supervisor.config import load_config
from llama_supervisor.manager import ServerManager
from llama_supervisor.models import list_models, resolve_model_path
from llama_supervisor.telemetry import TelemetryCollector


def create_app() -> FastAPI:
    config = load_config()
    manager = ServerManager(
        model_root=config.model_root,
        data_root=config.data_root,
        log_max_bytes=config.log_max_bytes,
        log_backups=config.log_backups,
    )
    telemetry = TelemetryCollector(
        interval=config.snapshot_interval,
        data_root=config.data_root,
        manager=manager,
    )
    telemetry.start()

    app = FastAPI()
    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    static_dir = base_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def require_token(request: Request) -> None:
        if not config.token:
            return
        header = request.headers.get("Authorization", "")
        expected = f"Bearer {config.token}"
        if header != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        return templates.TemplateResponse("models.html", {"request": request})

    @app.get("/servers", response_class=HTMLResponse)
    async def servers_page(request: Request):
        return templates.TemplateResponse("servers.html", {"request": request})

    @app.get("/api/models", dependencies=[Depends(require_token)])
    async def api_models() -> dict[str, Any]:
        items = list_models(config.model_root)
        return {
            "root": str(config.model_root),
            "models": [item.__dict__ for item in items],
        }

    @app.post("/api/servers/start", dependencies=[Depends(require_token)])
    async def api_start_server(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = manager.start_server(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"server": record.__dict__}

    @app.post("/api/servers/stop/{server_id}", dependencies=[Depends(require_token)])
    async def api_stop_server(server_id: str) -> dict[str, Any]:
        try:
            record = manager.stop_server(server_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"server": record.__dict__}

    @app.get("/api/servers", dependencies=[Depends(require_token)])
    async def api_servers() -> dict[str, Any]:
        records = manager.list_servers()
        return {"servers": [record.__dict__ for record in records]}

    @app.get("/api/servers/{server_id}/logs", dependencies=[Depends(require_token)])
    async def api_logs(server_id: str, tail: int = 2000) -> dict[str, Any]:
        tail = max(0, min(int(tail), 10000))
        record = manager.get_record(server_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown server id")
        lines = _tail_lines(Path(record.log_path), tail)
        return {"server_id": server_id, "tail": tail, "lines": lines}

    @app.get("/api/servers/{server_id}/logfile", dependencies=[Depends(require_token)])
    async def api_logfile(server_id: str) -> Response:
        record = manager.get_record(server_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown server id")
        log_path = Path(record.log_path)
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="log file not found")

        def _iter_logs():
            for path in _ordered_log_paths(log_path, config.log_backups):
                if path.exists():
                    yield path.read_text(encoding="utf-8")

        headers = {"Content-Disposition": f"attachment; filename={log_path.name}"}
        return StreamingResponse(_iter_logs(), headers=headers, media_type="text/plain")

    @app.get("/api/servers/{server_id}/metrics", dependencies=[Depends(require_token)])
    async def api_metrics(
        server_id: str,
        since: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        record = manager.get_record(server_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown server id")
        since_ts = _parse_since(since)
        limit = max(1, min(int(limit), 2000))
        path = Path(record.metrics_path)
        entries = _read_metrics(path, since_ts, limit)
        return {"server_id": server_id, "metrics": entries}

    @app.get("/api/servers/{server_id}/model", dependencies=[Depends(require_token)])
    async def api_model_path(server_id: str) -> dict[str, Any]:
        record = manager.get_record(server_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown server id")
        abs_path = resolve_model_path(config.model_root, record.model)
        return {"server_id": server_id, "model_path": str(abs_path)}

    @app.get("/api/health", dependencies=[Depends(require_token)])
    async def api_health() -> dict[str, Any]:
        return {"ok": True}

    return app


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def _ordered_log_paths(path: Path, backups: int) -> list[Path]:
    paths: list[Path] = []
    for idx in range(backups, 0, -1):
        paths.append(path.with_suffix(path.suffix + f".{idx}"))
    paths.append(path)
    return paths


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp()


def _read_metrics(path: Path, since: float | None, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                ts = payload.get("ts")
                if isinstance(ts, (int, float)) and ts < since:
                    continue
            entries.append(payload)
            if len(entries) > limit:
                entries = entries[-limit:]
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llama_supervisor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("uvicorn is required to run the service") from exc
    uvicorn.run("llama_supervisor.app:create_app", host=args.host, port=args.port, factory=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
