# Roadmap Steps

This file is a checklist of implemented steps and the expected “definition of done”.
Keep each step small and merged with tests.

---

## Step 0: Repo bootstrap (this PR)
- Create src-layout Python package `spectator` (Python 3.12)
- Add docs:
  - `docs/architecture.md` (high-level)
  - `docs/contracts.md` (this)
  - `docs/roadmap_steps.md`
  - `docs/testing.md`
- Add testing:
  - `pytest`, basic config
  - fake backend for tests
  - unit tests for:
    - NOTES extraction/parsing
    - TOOL_CALLS extraction/parsing
    - trace writer JSONL output
- Add minimal runtime skeleton:
  - `Checkpoint`, `State`, `ChatMessage`
  - checkpoint store: load/save latest (atomic write)
  - trace writer: JSONL to `./data/traces/`

Definition of done:
- `pytest -q` passes

---

## Step 7: Minimal RuntimeController + checkpoint store
- RuntimeController loads checkpoint at turn start; writes after turn end
- checkpoint store supports:
  - `load_latest(session_id)`
  - `save(session_id, checkpoint)` (atomic; backup optional)
- trace tail can be updated

Definition of done:
- smoke test loads/saves checkpoint without corruption

---

## Step 8: Multi-role pipeline + role specs
- Roles: reflection → planner → critic → governor
- Each role has spec:
  - name
  - system prompt
  - inference params
  - wants_retrieval flag (default False here)
- Pipeline accumulates upstream outputs
- Governor produces final response

Definition of done:
- test pipeline ordering with fake backend

---

## Step 9: Telemetry plumbing
- Telemetry snapshot object available per role
- Role may see telemetry based on role spec
- Trace telemetry per turn

Definition of done:
- unit test: telemetry injected when enabled

---

## Step 10: Deterministic condensation
- Bounded `recent_messages`, `open_loops`, `decisions`, etc.
- Action `condense_now` compacts state/messages deterministically

Definition of done:
- unit tests: condense reduces size and is deterministic

---

## Step 11: Smart condensation + writeback memory
- Vector store: SQLite embeddings store
- Embedder: hash embedder fallback (real embeddings optional later)
- `condense_smart`:
  - evict old messages
  - optional LLM summarizer when warn <= 1
  - write evicted raw + summary into vector DB
  - attach memory refs to state
  - then run deterministic condense

Definition of done:
- unit tests: writeback stores entries, refs updated, fallback works

---

## Step 12: Retrieval + injection + citations
- MemoryRetriever queries vector DB by embedding
- Retrieved memory injected into system prompt for roles with wants_retrieval=True
- Trace which memory IDs were used

Definition of done:
- tests: retrieval formatting includes IDs; injection works

---

## Step 13: Tool execution framework
- TOOL_CALLS parsing
- Tool registry + executor with sandbox + permissioning:
  - fs tools
  - shell.exec allowlist + deny list
- Governor tool loop:
  - Governor pass 1 may request tools
  - execute tools
  - Governor pass 2 sees tool results and outputs final answer
- Trace tool_plan/tool_start/tool_done

Definition of done:
- tests: permission denies dangerous commands; sandbox prevents path escape

---

## Step 14: Controlled web browsing
- `http.get` tool with:
  - capability gating (`net`, `net:<domain>`)
  - optional global allowlist
  - SQLite cache + TTL
  - byte/time caps
  - html->text stripping
- NOTES actions:
  - request_permission:*
  - grant_permission:*
- Expose pending/granted caps in state block for UI/admin
- Add admin CLI:
  - `scripts/capabilities_admin.py` list/grant/revoke/clear-pending

Definition of done:
- tests: URL permission checks, cache hit/miss behavior with mocked http transport
