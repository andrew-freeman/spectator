from __future__ import annotations

import time
from dataclasses import asdict

from spectator.core.tracing import TraceEvent, TraceWriter
from spectator.core.types import ChatMessage, State
from spectator.runtime.capabilities import apply_permission_actions
from spectator.runtime.checkpoints import load_or_create, save_checkpoint
from spectator.runtime.notes import NotesPatch, extract_notes


def _apply_notes_patch(state: State, patch: NotesPatch) -> None:
    if patch.set_goals:
        state.goals = list(patch.set_goals)
    if patch.add_open_loops:
        state.open_loops.extend(patch.add_open_loops)
    if patch.close_open_loops:
        state.open_loops = [loop for loop in state.open_loops if loop not in patch.close_open_loops]
    if patch.add_decisions:
        state.decisions.extend(patch.add_decisions)
    if patch.add_constraints:
        state.constraints.extend(patch.add_constraints)
    if patch.set_episode_summary is not None:
        state.episode_summary = patch.set_episode_summary
    if patch.add_memory_tags:
        state.memory_tags.extend(patch.add_memory_tags)


def run_turn(session_id: str, user_text: str, backend) -> str:
    checkpoint = load_or_create(session_id)
    checkpoint.recent_messages.append(ChatMessage(role="user", content=user_text))

    prompt = user_text
    writer = TraceWriter(session_id)
    writer.write(TraceEvent(ts=time.time(), kind="llm_req", data={"prompt": prompt}))
    assistant_text = backend.complete(prompt)
    writer.write(TraceEvent(ts=time.time(), kind="llm_done", data={"response": assistant_text}))

    visible_text, patch = extract_notes(assistant_text)
    if patch is not None:
        _apply_notes_patch(checkpoint.state, patch)
        if patch.actions:
            action_report = apply_permission_actions(checkpoint.state, patch.actions)
            writer.write(
                TraceEvent(
                    ts=time.time(),
                    kind="actions",
                    data={"role": "assistant", "actions": patch.actions, **action_report},
                )
            )
        writer.write(
            TraceEvent(
                ts=time.time(),
                kind="notes_patch",
                data=asdict(patch),
            )
        )

    checkpoint.recent_messages.append(ChatMessage(role="assistant", content=visible_text))
    if writer.path.exists():
        checkpoint.trace_tail = [writer.path.name]
    save_checkpoint(checkpoint)
    return visible_text
