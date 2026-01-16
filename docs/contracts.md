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
