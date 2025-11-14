# Architecture Overview

## System Description
This repository implements a deployable self-reflective hierarchical meta-cognitive
system composed of a base reasoning layer and a meta reasoning layer. The base
layer contains three collaborating agents:

- **Actor** – generates structured analysis, multi-step plans, and tool calls.
- **Critic** – audits the actor output for safety violations, contradictions,
  and missing context.
- **Governor** – deterministically arbitrates between actor and critic outputs,
  enforces safety policy, and schedules tool execution.

The meta layer operates every \(N\) cycles to adapt how the base layer reasons.
It introduces three higher-order agents:

- **Meta-Actor** – proposes strategic adjustments, parameter deltas, and
  procedural improvements for the base layer.
- **Meta-Critic** – stress-tests those proposals for coherence, stability, and
  safety.
- **Meta-Governor** – conservatively merges approved adjustments into the
  system configuration.

## Component Layout
```
app/
  actor/            # Actor prompts + runner
  critic/           # Critic prompts + runner
  governor/         # Deterministic arbitration helpers
  meta/             # Meta-layer prompts, runners, and governor
  api/              # FastAPI supervisor, state + tool managers
  models/           # OpenAI-compatible HTTP client + local llama.cpp client
  config/           # Cognitive parameter defaults and system limits
```

## Data Flow Summary
1. External caller submits objectives to the FastAPI supervisor.
2. The supervisor invokes the actor runner which builds a strict JSON prompt and
   forwards it to the configured LLM.
3. The critic runner reviews the actor output and returns structured feedback.
4. The governor merges or selects a plan following deterministic rules and
   triggers tool execution via the tool executor.
5. State transitions and tool results are logged by the state manager and may be
   persisted to memory.
6. Every \(N\) cycles the meta layer executes, proposing parameter deltas and
   updating `app/config/cog_params.json` when safe.

## Key Design Principles
- **Deterministic safety enforcement** – governor decisions follow explicit,
  reproducible rules derived from critic risk levels and confidence scores.
- **Strict JSON interfaces** – every prompt and response uses JSON schemas to
  simplify downstream parsing and validation.
- **Incremental meta-updates** – the meta governor clamps parameter changes to
  ±0.05 per cycle and only applies them with critic approval.
- **Tool abstraction** – all tools are described using OpenAI-compatible
  function schemas so the actor can trigger deterministic side effects.
