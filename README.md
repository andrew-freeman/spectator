# Spectator

**A deterministic, inspectable runtime for structured LLM reasoning**

Spectator is a lightweight framework for running large language models through a **fixed, multi-role reasoning pipeline** with **explicit state**, **strict contracts**, and **full traceability**.

It is designed for engineers who want **predictable behavior**, **auditable outputs**, and **clear system boundaries**—not opaque “agent magic”.

---

## Why Spectator exists

Most LLM applications blur together:

* prompts
* hidden chain-of-thought
* implicit memory
* tool execution
* state mutation

This makes systems hard to debug, hard to trust, and impossible to reason about formally.

**Spectator separates concerns.**

It enforces a clear boundary between:

* *reasoning* vs *state*
* *instructions* vs *data*
* *visible output* vs *internal scaffolding*
* *model behavior* vs *runtime guarantees*

---

## Core idea

Every user turn is processed through a **fixed pipeline**:

```
Reflection → Planner → Critic → Governor
```

Each role:

* Receives the same structured inputs
* Produces a single text output
* May optionally emit **structured NOTES** to update state

Only the **Governor** produces the final user-visible response.

---

## Key guarantees (v1.x)

Spectator v1.x provides strong, explicit guarantees:

### 🧠 Deterministic structure

* Fixed role order
* No dynamic role creation
* No recursive loops

### 🧾 Explicit state

* All long-lived state is stored in a typed `State` object
* State changes **only** via structured `NOTES_JSON`
* No implicit memory writes

### 🧼 Sanitized outputs

* Internal scaffolding (`STATE`, `HISTORY`, `UPSTREAM`, tool blocks, etc.) is **never shown to users**
* Reasoning delimiters are stripped
* Visible output is post-processed and audited

### 📜 History treated as data

* Conversation history is **read-only context**
* Instructions inside history are ignored
* No “instruction persistence” hacks

### 🛠️ Controlled tool use

* Tools must be explicitly requested via structured blocks
* Tool execution is permission-gated
* Results are reinjected deterministically

### 🔍 Full observability

* Every LLM request and response is traced
* Sanitization, tool calls, and state changes are logged
* JSONL traces enable offline analysis and replay

---

## What Spectator is *not*

Spectator is intentionally **not**:

* A self-modifying agent
* A long-horizon autonomous system
* A personality engine
* A hidden chain-of-thought extractor
* A framework that “just lets the model decide”

Those designs may be valid—but Spectator prioritizes **control, safety, and inspectability**.

---

## Supported backends

Spectator is backend-agnostic by design.

Currently supported:

* **llama-server** (local inference via OpenAI-compatible API)
* **Fake backend** (for tests)

Backends are adapters; the **runtime contracts remain the same** regardless of model.

---

## Typical use cases

Spectator is well-suited for:

* Safety-critical reasoning systems
* Tool-using assistants with strict boundaries
* Research on prompt robustness and prompt injection
* Model behavior evaluation and comparison
* Long-running local LLM experiments
* Systems that must be debugged months later

---

## Example

```bash
python -m spectator run --backend llama --text "Summarize the last three user messages."
```

Behind the scenes:

* History is formatted and bounded
* Each role runs independently
* State is preserved via checkpoints
* Output is sanitized and traced

You can inspect everything afterward.

---

## Project status

* **Architecture version:** v1.0 (frozen)
* **API stability:** Stable for v1.x
* **Test coverage:** Extensive (unit + integration + adversarial)
* **Focus:** Correctness > cleverness

Future work (v2.0+) will be clearly separated from v1.x stability.

---

## Admin/debug UI

Spectator includes a minimal FastAPI-based admin UI for inspecting JSONL traces.

```bash
uvicorn spectator.admin.app:create_app --factory --port 8000
```

Set `DATA_ROOT` to point at your data directory, and optionally set
`SPECTATOR_ADMIN_TOKEN` to require an `X-Admin-Token` header for access.

To run the admin tests locally:

```bash
pip install -e ".[admin]" && pytest -m admin
```

Open loops endpoints (task tracking):

* `GET /api/sessions/{session_id}/open_loops`
* `POST /api/sessions/{session_id}/open_loops`
* `POST /api/sessions/{session_id}/open_loops/{loop_id}/close`

---

## Design philosophy

> **Stability is a feature.**

Spectator prefers:

* explicit contracts over flexibility
* boring correctness over surprising intelligence
* debuggability over clever hacks

If a system cannot explain *why* it behaved the way it did, it is not finished.

---

## License

MIT License.
Use it, modify it, and build on it—just don’t pretend it does something it doesn’t.

---

## Running with llama-server

1. Start llama-server (example):

   ```bash
   llama-server --model /path/to/model.gguf --port 8080
   ```

2. Run Spectator with the llama backend:

   ```bash
   export SPECTATOR_BACKEND=llama
   export LLAMA_SERVER_BASE_URL=http://127.0.0.1:8080
   # Optional: export LLAMA_SERVER_MODEL=your-model-name
   python -m spectator run --text "Hello"
   ```
