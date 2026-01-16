# Architecture

Spectator is a lightweight runtime that wraps an LLM pipeline with stable data
contracts, structured parsing, and trace capture. The core package uses a
src-layout (`src/spectator`) and keeps the runtime surface small so that future
steps can layer in checkpoint storage, role pipelines, and tool execution
without breaking existing interfaces.

Key layers:
- `core`: immutable data contracts (`State`, `Checkpoint`, `ChatMessage`) and
  tracing utilities for JSONL event streams.
- `runtime`: parsing utilities for structured model output blocks (NOTES and
  TOOL_CALLS).
- `backends`: minimal backend interfaces plus a fake backend for tests.

Artifacts:
- Traces are written as JSONL under `./data/traces/<session_id>.jsonl`.
- Tests live under `tests/` and validate parsing and tracing behavior.
