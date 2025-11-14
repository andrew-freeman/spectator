# Deployment Instructions

Follow these steps to run the self-reflective LLM mind locally or in a
production environment.

## 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional backends:
- `pip install llama-cpp-python` for local GGUF models (e.g., Llama 3.1 70B or
  Qwen 2.5 72B via quantised variants).
- `pip install openai` if you prefer the official SDK (the provided HTTP client
  works with raw API keys).

## 2. Configure Model Clients
Choose one of the supported backends and prepare credentials:

### OpenAI-Compatible Endpoint
```python
from app.api.main import configure_supervisor
from app.models.openai_client import OpenAIClient

client = OpenAIClient(model="gpt-4o-mini", api_key="YOUR_KEY", base_url="https://api.openai.com/v1")
configure_supervisor(actor_client=client)
```

To use hosted open-source models (e.g., Groq, Together, Fireworks), set
`base_url` to the provider endpoint.

### Local Llama/Qwen Model
```python
from app.api.main import configure_supervisor
from app.models.local_llm_client import LocalLLMClient

client = LocalLLMClient(model_path="/path/to/llama-3.1-70b.gguf")
configure_supervisor(actor_client=client)
```

## 3. Launch the API
```bash
uvicorn app.api.main:app --reload
```

The server exposes:
- `GET /health` – liveness probe.
- `POST /run-cycle` – run one reasoning cycle.
- `GET /history` – retrieve logged cycles.

## 4. Execute Reasoning Cycles
Send JSON payloads via `POST /run-cycle`:
```json
{
  "objectives": ["Diagnose thermal spike"],
  "context": {"sensors": {"temp": 78.4}},
  "memory": ["Fan speed previously capped at 0.5"]
}
```

Responses include actor, critic, governor, tool results, and optional meta layer
updates when the cycle count hits `meta_frequency`.

## 5. Persisted Configuration
- `app/config/cog_params.json` – updated by the meta governor with bounded
  parameter changes.
- `app/config/system_limits.json` – static constraints such as maximum tool calls
  or temperature bounds.

Ensure the application has write access to these files when running in
containers or serverless environments.
