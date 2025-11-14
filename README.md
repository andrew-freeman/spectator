# self-reflective-llm-mind

A deployable, self-reflective hierarchical meta-cognitive system that combines
an actor/critic/governor base layer with a meta layer that tunes cognitive
parameters over time. The system supports OpenAI-compatible APIs and local
llama.cpp deployments of Llama 3.1 70B, Qwen 2.5 72B, or similar models.

## Features
- Structured prompts that force JSON outputs for deterministic parsing.
- Deterministic governor arbitration with explicit safety rules.
- Meta reasoning loop that clamps parameter deltas to ±0.05 per update.
- FastAPI supervisor providing `/health`, `/run-cycle`, and `/history` endpoints.
- Tool execution layer exposing OpenAI-style function schemas for sensors,
  actuators, and memory utilities.

## Repository Layout
```
app/
  actor/      # Actor prompt + runner
  critic/     # Critic prompt + runner
  governor/   # Arbitration logic and helpers
  meta/       # Meta-layer prompts, runners, and governor logic
  api/        # FastAPI supervisor, tool schemas, state + memory managers
  models/     # OpenAI-compatible HTTP client and llama.cpp local client
  config/     # Cognitive parameters + system limits (persisted between runs)
```

See `app/docs/architecture.md` for a detailed component breakdown.

## Quickstart
1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure an LLM client**

   *OpenAI-compatible endpoint*
   ```python
   from app.api.main import configure_supervisor
   from app.models.openai_client import OpenAIClient

   client = OpenAIClient(model="gpt-4o-mini", api_key="YOUR_KEY")
   configure_supervisor(actor_client=client)
   ```

   *Local llama.cpp model*
   ```python
   from app.api.main import configure_supervisor
   from app.models.local_llm_client import LocalLLMClient

   client = LocalLLMClient(model_path="/models/llama-3.1-70b.gguf")
   configure_supervisor(actor_client=client)
   ```

3. **Run the API**
   ```bash
   uvicorn app.api.main:app --reload
   ```

4. **Invoke a reasoning cycle**
   ```bash
   curl -X POST http://localhost:8000/run-cycle \
     -H "Content-Type: application/json" \
     -d '{"objectives": ["Stabilise environment"], "context": {"sensors": {"temp": 77.1}}}'
   ```

5. **Inspect history**
   ```bash
   curl http://localhost:8000/history
   ```

## Configuration Files
- `app/config/cog_params.json` – runtime parameters updated by the meta-governor.
- `app/config/system_limits.json` – guardrails such as maximum tool calls or
  bounded fan speed values.

The meta layer runs every `meta_frequency` cycles (default `3`) and persists
updates back into `cog_params.json` to influence subsequent actor/critic
behaviour.

## Documentation
Additional documentation lives in `app/docs/`:
- `architecture.md` – component map and system description.
- `reasoning_flow.md` – end-to-end reasoning cycle walkthrough.
- `meta_layer_design.md` – detailed meta layer schema and safeguards.
- `deployment_instructions.md` – step-by-step setup and operation guide.

## License
This repository is intended for research and experimentation. Integrate
responsibly and follow provider-specific terms of service for hosted models.
