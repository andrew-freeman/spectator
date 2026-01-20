# Llama Supervisor

Standalone FastAPI service to launch and monitor `llama-server` instances with basic telemetry and log retention.

## Run

```bash
python -m llama_supervisor.app --host 0.0.0.0 --port 9000
```

## Config

Environment variables:

- `MODEL_ROOT` (default `/nvme_mod/models/`)
- `SUPERVISOR_DATA_ROOT` (default `data`)
- `SUPERVISOR_TOKEN` (optional, if set requires `Authorization: Bearer <token>` on `/api/*`)
- `SNAPSHOT_INTERVAL` (default `2.0`)
- `LOG_MAX_BYTES` (default `209715200`)
- `LOG_BACKUPS` (default `3`)

You can also update the model root from the web UI; it persists to `data/llama_supervisor.json`.

## Example start request

```bash
curl -X POST http://localhost:9000/api/servers/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "gpu": 0,
    "port": 8080,
    "host": "0.0.0.0",
    "model": "openai_gpt-oss-20b-Q6_K_L.gguf",
    "ngl": 99,
    "verbose": true
  }'
```

## systemd example

```ini
[Unit]
Description=Llama Supervisor
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/sp-ng
Environment=MODEL_ROOT=/nvme_mod/models/
Environment=SUPERVISOR_DATA_ROOT=/path/to/sp-ng/data
ExecStart=/path/to/sp-ng/.venv/bin/python -m llama_supervisor.app --host 0.0.0.0 --port 9000
Restart=always

[Install]
WantedBy=multi-user.target
```
