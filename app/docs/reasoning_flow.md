# Reasoning Flow

The supervisor orchestrates a deterministic cycle across the actor, critic, and
governor components. The process repeats for each call to the `/run-cycle`
endpoint.

## Base Layer Sequence
1. **Input preparation** – Objectives, context, and relevant memory snippets are
   fed into the actor prompt builder to produce a structured JSON-only prompt.
2. **Actor reasoning** – The configured LLM responds with analysis, plan, and
   tool call proposals adhering to the schema defined in
   `app/actor/actor_prompt.py`.
3. **Critic review** – The critic prompt inspects the actor payload, returns a
   risk level, enumerates detected issues, and emits recommendations.
4. **Governor arbitration** – `app/governor/governor_logic.py` combines the two
   outputs using deterministic rules:
   - *Unsafe risk* ⇒ trust the critic and halt tool execution.
   - *Low critic confidence* ⇒ proceed with the actor plan.
   - *Partial mismatch* ⇒ merge actor plan with critic recommendations.
   - *Incomplete actor data* ⇒ request more information.
5. **Tool execution** – When permitted, the governor forwards tool calls to the
   executor which interacts with state and memory managers.
6. **Logging** – The state manager records cycle metadata, tool results, and
   updated shared state.

## Meta Layer Invocation
Every \(N\) cycles, defined by `meta_frequency` in `cog_params.json`, the supervisor
runs the meta reasoning loop:

1. **Meta-Actor** analyses recent decisions and suggests cognitive strategy
   changes plus bounded parameter adjustments.
2. **Meta-Critic** validates the proposal for safety, stability, and coherence.
3. **Meta-Governor** clamps deltas to ±0.05, applies safe changes, updates
   `cog_params.json`, and records any notes for future cycles.

This meta layer ensures the base reasoning loop adapts gradually without losing
stability or violating configured system limits.
