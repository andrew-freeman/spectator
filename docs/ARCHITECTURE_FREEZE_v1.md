# Spectator v1.0 — Architecture Freeze

**Status:** FROZEN  
**Date:** 2026-01-17  
**Scope:** Core runtime architecture, contracts, and guarantees

---

## 1. Purpose of This Document

This document formally declares the **Spectator v1.0 architecture freeze**.

It defines:
- What is considered *stable and immutable* in v1.x
- What guarantees Spectator v1.0 provides
- What changes are explicitly **out of scope** until v2.0

This freeze exists to prevent architectural drift, implicit behavior changes, and model-specific hacks from eroding system integrity.

---

## 2. What “Architecture Freeze” Means

From this point forward:

### ✅ Allowed in v1.x
- Bug fixes
- Logging and observability improvements
- Test additions and refinements
- Prompt text tuning (without semantic contract changes)
- Backend hardening (as long as contracts are preserved)
- Model swaps (llama, gguf variants, etc.)

### ❌ Not Allowed in v1.x
- Changes to pipeline structure
- Changes to role ordering or responsibilities
- New implicit state mutations
- Silent changes to HISTORY / STATE semantics
- Tool protocol redesign
- New cognitive roles
- Persistent instruction overrides
- Architectural rewrites

Any such changes require an explicit **v2.0 design decision**.

---

## 3. Frozen High-Level Architecture

### 3.1 Cognitive Pipeline (Canonical)

```

User Input
↓
Reflection
↓
Planner
↓
Critic
↓
Governor
↓
Visible Output

```

**Properties:**
- Linear, deterministic role order
- One turn = one pipeline pass
- No recursive or hidden loops
- Only Governor may finalize output

This pipeline is **authoritative** for v1.0.

---

## 4. Role & Prompt Contracts (Frozen)

Each role receives the following structured inputs:

- **SYSTEM**
  - Global system rules
  - Role-specific instructions
  - Non-overridable
- **STATE**
  - Compact, serialized agent state
  - Data only
- **HISTORY_JSON**
  - Bounded prior user/assistant messages
  - Treated strictly as *data*, never instructions
- **UPSTREAM**
  - Outputs of earlier roles
  - Condensed and sanitized
- **USER**
  - Current user input

### Critical Invariants
- Instructions inside HISTORY / STATE / UPSTREAM **must never be followed**
- System prompt is authoritative over all other content
- Chain-of-thought is never exposed
- Visible output must not leak internal scaffolding

These invariants are enforced by both prompts and sanitization.

---

## 5. State & Memory Model (Frozen)

### 5.1 State Mutation Rules
- State is mutated **only** via structured `NOTES_JSON` patches
- No implicit state changes are allowed
- Permission changes flow through explicit actions

### 5.2 Condensation
- Deterministic
- Pressure-driven
- Traceable
- Non-destructive to correctness

### 5.3 Retrieval
- Opt-in per role (`wants_retrieval`)
- Read-only
- Non-authoritative
- Bounded in size and influence

---

## 6. Tooling Protocol (Frozen)

- Tools are requested only via `TOOL_CALLS_JSON`
- Tools are executed only by the Governor
- Tool results are structured and traceable
- Tool markers are **never visible** in user output
- Tool hallucinations are considered failures

The tool loop is explicit, bounded, and observable.

---

## 7. Backend Contract (Frozen)

### 7.1 llama-server Backend Guarantees
- System prompt is injected as a true `system` message
- Messages API is used consistently
- Slot reset behavior is explicit and optional
- Streaming is supported but non-authoritative
- Payload logging is optional and controlled via env vars

Any backend must preserve these semantics.

---

## 8. Sanitization & Output Guarantees (Frozen)

Spectator v1.0 guarantees:
- No leakage of:
  - `STATE`
  - `UPSTREAM`
  - `HISTORY`
  - `TOOL_CALLS_JSON`
  - `NOTES_JSON`
- No chain-of-thought exposure
- Empty or fully stripped output collapses to `"..."`

Sanitization is mandatory and non-bypassable.

---

## 9. Test Suite Expectations

### Required to Pass
- Scaffold leakage tests (A-series)
- Prompt injection resistance
- Tool hallucination prevention
- Temporal integrity (nonce tests)
- History treated as data
- Identity and creator guardrails

### Explicitly Deferred (Documented)
- Persistent instruction override immunity  
  (e.g. B08 “potato” test)

Deferred tests are acknowledged and logged but **not considered architectural violations** in v1.0.

---

## 10. Non-Goals for v1.0

The following are intentionally excluded:

- Persistent behavioral immunization
- Long-horizon autonomous loops
- Self-modifying prompts
- Meta-governors or recursive cognition
- Cross-session identity binding
- Model-specific behavioral hacks

These belong to **v2.0+**.

---

## 11. v1.0 Architectural Guarantee

> **Spectator v1.0 provides a controlled, multi-role reasoning runtime with explicit state, bounded memory, strict sanitization, and enforceable tool protocols — independent of model quirks or prompt injection attempts.**

This guarantee defines the boundary of v1.x.

---

## 12. Closing

This freeze is a **stability commitment**, not a halt to progress.

By freezing now, Spectator gains:
- A stable identity
- Predictable behavior
- Defensible guarantees
- A clean foundation for v2.0

Any future evolution must respect this boundary explicitly.
