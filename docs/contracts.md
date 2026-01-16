# Spectator Contracts

This document defines the stable interfaces and data contracts used across the system.
Keep it updated whenever adding a new tool, event type, state field, or wire protocol.

---

## 1) Core data types

### 1.1 State

`State` is the persisted long-lived agent state stored in the checkpoint.

Required fields:

- `goals: list[str]`
- `open_loops: list[str]`
- `decisions: list[str]`
- `constraints: list[str]`
- `episode_summary: str`
- `memory_tags: list[str]`
- `memory_refs: list[str]`

Step 14 capability gating fields:

- `capabilities_granted: list[str]`  
  Examples:
  - `"net"` (broad network permission, still constrained by policy allowlist if enabled)
  - `"net:huggingface.co"` (domain-scoped)
- `capabilities_pending: list[str]`  
  Requested capabilities awaiting approval (UI/admin).

Constraints:
- Lists contain strings only.
- `capabilities_pending` MUST NOT include anything already granted.

### 1.2 Checkpoint

`Checkpoint` is the persisted unit written to disk.

Fields:
- `session_id: str`
- `revision: int`
- `updated_ts: float`
- `state: State`
- `recent_messages: list[ChatMessage]`  (bounded by condensation)
- `trace_tail: list[str]` (optional; last N trace ids or filenames)

### 1.3 ChatMessage

Minimal contract:
- `role: str` in `{"system","user","assistant","tool"}`
- `content: str`

---

## 2) Structured NOTES patch protocol

The model may emit a NOTES JSON block to request state updates and actions.

Markers:
- Begin: `<<<NOTES_JSON>>>`
- End: `<<<END_NOTES_JSON>>>`

Payload format:

```json
{
  "set_goals": ["..."],
  "add_open_loops": ["..."],
  "close_open_loops": ["..."],
  "add_decisions": ["..."],
  "add_constraints": ["..."],
  "set_episode_summary": "...",
  "add_memory_tags": ["..."],
  "actions": ["condense_now", "condense_smart", "request_permission:net:huggingface.co"]
}
````

Rules:

* Absent keys = no change.
* `actions` are strings; unknown actions are ignored and traced.
* Permission actions:

  * `request_permission:<cap>` adds `<cap>` to `capabilities_pending` unless already granted.
  * `grant_permission:<cap>` adds `<cap>` to `capabilities_granted` and removes from pending.

The runtime must:

* extract NOTES patch from assistant content
* apply patch to `Checkpoint.state`
* run actions after patch apply (if any)
* trace patch and actions

---

## 3) Tool call protocol (Step 13)

The model may request tool execution via a tool calls JSON block.

Markers:

* Begin: `<<<TOOL_CALLS_JSON>>>`
* End: `<<<END_TOOL_CALLS_JSON>>>`

Payload: JSON array

```json
[
  {"id":"t1","tool":"fs.list_dir","args":{"path":"."}},
  {"id":"t2","tool":"shell.exec","args":{"cmd":"ls -la","timeout_s":10}}
]
```

Fields:

* `id: str` required; unique within the response
* `tool: str` required; namespaced string
* `args: object` required

The runtime must:

* parse and remove tool-call block from user-visible text
* execute tools in order with permissioning
* append results as tool-role messages to the next LLM call in the tool loop

Tool result message (role="tool") content is JSON:

```json
{
  "id": "t1",
  "tool": "fs.list_dir",
  "ok": true,
  "output": { ... },
  "error": null
}
```

---

## 4) Tool registry (Step 13/14)

Baseline tools:

* `fs.read_text {path:str, max_bytes:int=20000}`
* `fs.list_dir {path:str=".", max_entries:int=200}`
* `fs.write_text {path:str, text:str, overwrite:bool=false}`
* `shell.exec {cmd:str, timeout_s:int=20}`
* `http.get {url:str, use_cache:bool=true}`

Sandbox rules:

* Filesystem tools operate under a sandbox root; deny path escapes.
* Shell tool runs in a sandbox cwd and is allowlisted; deny dangerous substrings.

Network rules (Step 14):

* `http.get` is denied unless capability allows it:

  * allowed if `capabilities_granted` contains `"net"` or `"net:<domain>"`
  * if global allowlist is configured, `"net"` allows only allowlisted domains; `"net:<domain>"` overrides allowlist

Caching:

* `http.get` uses SQLite cache with TTL; returns cached response if fresh.

---

## 5) Retrieval (Step 12)

Memory retrieval returns items containing stable memory IDs.

Retrieved block injected into system prompt:

```
=== RETRIEVED_MEMORY ===
- id=mem_xxx score=0.412
  excerpt...
=== END_RETRIEVED_MEMORY ===
```

Optional user-facing citations:

* LLM may cite `[mem:<id>]` inline when using retrieved content.

---

## 6) Trace events

Traces are written as JSONL (one event per line) with:

* `ts: float` unix timestamp
* `kind: str`
* `data: object`

Required kinds:

* `llm_req`
* `llm_stream`
* `llm_done`
* `notes_patch`
* `actions`
* `retrieval`
* `tool_plan`
* `tool_start`
* `tool_done`

---

## 7) Worker model & backends

Backend interface:

* `chat(request) -> response`
* `stream_chat(request, cancel) -> async iterator of deltas`

The runtime controller is resilient:

* tools and web fetch do not crash main loop; failures become tool results with ok=false
