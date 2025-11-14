# Meta Layer Design

The meta reasoning stack tunes system behaviour without compromising stability.
Each component emits JSON structures that map directly to configuration updates.

## Meta-Actor Schema
The meta-actor prompt requires the following fields:

- `meta_thoughts` – Observations about base-layer performance.
- `cognitive_strategy` – Desired priorities and cadence tweaks for upcoming
  cycles.
- `parameter_adjustments` – Map of parameter names to `{delta, justification}`
  pairs. Deltas must remain within ±0.05 per cycle.
- `meta_improvements` – Optional process upgrades or tooling suggestions.
- `assumptions` – Explicit statements backing the proposed adjustments.

## Meta-Critic Responsibilities
The meta-critic verifies that the proposed adjustments:

1. Respect system limits and the ±0.05 delta rule.
2. Maintain stability across cycles (e.g., no oscillating thresholds).
3. Include sufficient justification for each change.
4. Provide alternative improvements when rejecting risky proposals.

It returns `meta_issues`, a `risk_rating`, a confidence score, and additional
`meta_improvements` for future iterations.

## Meta-Governor Algorithm
`app/meta/meta_governor_logic.py` enforces conservative adoption:

1. Reject proposals immediately when `risk_rating` is `unsafe` or `high`.
2. Request revisions when the meta-critic has low confidence (<0.5) and reports
   outstanding issues.
3. Otherwise clamp each delta to ±0.05, update the in-memory configuration, and
   persist the new `cog_params.json` snapshot.

The meta governor also records notes about approved or rejected changes so the
supervisor can expose them in API responses.
