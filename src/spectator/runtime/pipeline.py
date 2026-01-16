from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from spectator.core.telemetry import TelemetrySnapshot, collect_basic_telemetry
from spectator.core.tracing import TraceEvent
from spectator.core.types import Checkpoint, State
from spectator.runtime.condense import CondensePolicy, condense_state, condense_upstream
from spectator.runtime.memory_feedback import compute_memory_pressure, format_memory_feedback
from spectator.runtime.notes import NotesPatch, extract_notes


@dataclass(slots=True)
class RoleSpec:
    name: str
    system_prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    wants_retrieval: bool = False
    telemetry: str = "none"
    memory_feedback: str = "none"


@dataclass(slots=True)
class RoleResult:
    role: str
    text: str
    notes: NotesPatch | None


def _compact_state(state: State) -> str:
    payload = asdict(state)
    return "{" + ",".join(f"{key}:{value!r}" for key, value in payload.items()) + "}"


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _apply_notes_patch(state: State, patch: NotesPatch) -> None:
    if patch.set_goals:
        state.goals = list(patch.set_goals)
    if patch.add_open_loops:
        _extend_unique(state.open_loops, patch.add_open_loops)
    if patch.close_open_loops:
        state.open_loops = [loop for loop in state.open_loops if loop not in patch.close_open_loops]
    if patch.add_decisions:
        _extend_unique(state.decisions, patch.add_decisions)
    if patch.add_constraints:
        _extend_unique(state.constraints, patch.add_constraints)
    if patch.set_episode_summary is not None:
        state.episode_summary = patch.set_episode_summary
    if patch.add_memory_tags:
        _extend_unique(state.memory_tags, patch.add_memory_tags)


def _compose_prompt(
    role: RoleSpec,
    state: State,
    upstream: list[RoleResult],
    user_text: str,
    telemetry: TelemetrySnapshot | None,
    memory_feedback: str | None,
    retrieval_block: str | None,
) -> str:
    parts = [role.system_prompt, f"STATE:\n{_compact_state(state)}"]
    if telemetry is not None and role.telemetry == "basic":
        telemetry_text = "\n".join(
            [
                "=== TELEMETRY (basic) ===",
                f"ts: {telemetry.ts}",
                f"pid: {telemetry.pid}",
                f"platform: {telemetry.platform}",
                f"python: {telemetry.python}",
                f"ram_total_mb: {telemetry.ram_total_mb}",
                f"ram_avail_mb: {telemetry.ram_avail_mb}",
                "=== END TELEMETRY ===",
            ]
        )
        parts.append(telemetry_text)
    if memory_feedback:
        parts.append(memory_feedback)
    if retrieval_block and role.wants_retrieval:
        parts.append(retrieval_block)
    if upstream:
        upstream_text = "\n".join(f"{result.role}: {result.text}" for result in upstream)
        parts.append(f"UPSTREAM:\n{upstream_text}")
    parts.append(f"USER:\n{user_text}")
    return "\n\n".join(part for part in parts if part)


def run_pipeline(
    checkpoint: Checkpoint,
    user_text: str,
    roles: Iterable[RoleSpec],
    backend,
    memory: MemoryContext | None = None,
    tracer=None,
) -> tuple[str, list[RoleResult], Checkpoint]:
    results: list[RoleResult] = []
    condense_policy = CondensePolicy()
    role_list = list(roles)
    telemetry_roles = [role.name for role in role_list if role.telemetry == "basic"]
    memory_feedback_roles = [role.name for role in role_list if role.memory_feedback == "basic"]
    telemetry_snapshot = collect_basic_telemetry() if telemetry_roles else None
    last_report = None
    memory_pressure_traced = False
    retrieval_block = None
    retrieval_roles = [role.name for role in role_list if role.wants_retrieval]
    if memory is not None and retrieval_roles:
        from spectator.memory.retrieval import format_retrieval_block, retrieve

        query_text = f"{user_text}\nSTATE:{_compact_state(checkpoint.state)}"
        retrieval_results = retrieve(query_text, memory.store, memory.embedder, top_k=5)
        retrieval_block = format_retrieval_block(retrieval_results)
        if tracer is not None:
            tracer.write(
                TraceEvent(
                    ts=time.time(),
                    kind="retrieval",
                    data={
                        "roles": retrieval_roles,
                        "k": 5,
                        "count": len(retrieval_results),
                        "ids": [record.id for record, _score in retrieval_results],
                        "scores": [score for _record, score in retrieval_results],
                    },
                )
            )
    if tracer is not None and telemetry_snapshot is not None:
        tracer.write(
            TraceEvent(
                ts=time.time(),
                kind="telemetry",
                data={
                    "roles": telemetry_roles,
                    "snapshot": asdict(telemetry_snapshot),
                },
            )
        )

    for role in role_list:
        if results:
            before_lengths = [len(result.text) for result in results]
            condensed_upstream = condense_upstream(results, condense_policy)
            after_lengths = [len(result.text) for result in condensed_upstream]
            if tracer is not None and sum(after_lengths) < sum(before_lengths):
                tracer.write(
                    TraceEvent(
                        ts=time.time(),
                        kind="condense",
                        data={
                            "scope": "upstream",
                            "role": role.name,
                            "report": {
                                "before_total_chars": sum(before_lengths),
                                "after_total_chars": sum(after_lengths),
                                "per_role": [
                                    {
                                        "role": result.role,
                                        "before_chars": before_len,
                                        "after_chars": after_len,
                                    }
                                    for result, before_len, after_len in zip(
                                        results, before_lengths, after_lengths
                                    )
                                ],
                            },
                        },
                    )
                )
            results = condensed_upstream

        pressure = compute_memory_pressure(checkpoint.state, condense_policy, results, last_report)
        if tracer is not None and memory_feedback_roles and not memory_pressure_traced:
            tracer.write(
                TraceEvent(
                    ts=time.time(),
                    kind="memory_pressure",
                    data={
                        "roles": memory_feedback_roles,
                        "ratios": {
                            "goals": pressure.goals_ratio,
                            "open_loops": pressure.open_loops_ratio,
                            "decisions": pressure.decisions_ratio,
                            "constraints": pressure.constraints_ratio,
                            "memory_tags": pressure.memory_tags_ratio,
                            "upstream": pressure.upstream_ratio,
                        },
                        "high_fields": pressure.high_fields,
                        "condensed": pressure.condensed,
                    },
                )
            )
            memory_pressure_traced = True
        memory_feedback_block = (
            format_memory_feedback(pressure) if role.memory_feedback == "basic" else None
        )
        prompt = _compose_prompt(
            role,
            checkpoint.state,
            results,
            user_text,
            telemetry_snapshot,
            memory_feedback_block,
            retrieval_block,
        )
        params = dict(role.params)
        params.setdefault("role", role.name)
        if tracer is not None:
            tracer.write(
                TraceEvent(
                    ts=time.time(),
                    kind="llm_req",
                    data={"role": role.name, "prompt": prompt},
                )
            )
        response = backend.complete(prompt, params=params)
        if tracer is not None:
            tracer.write(
                TraceEvent(
                    ts=time.time(),
                    kind="llm_done",
                    data={"role": role.name, "response": response},
                )
            )

        visible_text, patch = extract_notes(response)
        if patch is not None:
            _apply_notes_patch(checkpoint.state, patch)
            report = condense_state(checkpoint.state, condense_policy)
            last_report = report if report.trimmed else None
            if tracer is not None:
                tracer.write(
                    TraceEvent(
                        ts=time.time(),
                        kind="notes_patch",
                        data={"role": role.name, **asdict(patch)},
                    )
                )
                if report.trimmed:
                    tracer.write(
                        TraceEvent(
                            ts=time.time(),
                            kind="condense",
                            data={
                                "scope": "state",
                                "role": role.name,
                                "report": asdict(report),
                            },
                        )
                    )
        results.append(RoleResult(role=role.name, text=visible_text, notes=patch))

    final_text = results[-1].text if results else ""
    return final_text, results, checkpoint
