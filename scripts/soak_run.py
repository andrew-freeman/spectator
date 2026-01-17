from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.runtime.checkpoints import load_or_create, save_checkpoint
from spectator.runtime.condense import CondensePolicy
from spectator.runtime.notes import END_MARKER as NOTES_END_MARKER
from spectator.runtime.notes import START_MARKER as NOTES_START_MARKER
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.runtime.tool_calls import END_MARKER as TOOL_END_MARKER
from spectator.runtime.tool_calls import START_MARKER as TOOL_START_MARKER
from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler, read_text_handler
from spectator.tools.registry import ToolRegistry

MAX_MEMORY_REFS = 128
MAX_CAPABILITIES = 64
MAX_EPISODE_SUMMARY_CHARS = 2000


@dataclass
class SoakMetrics:
    turns: int = 0
    trace_file_size: int = 0
    checkpoint_size: int = 0
    max_open_loops: int = 0
    max_memory_tags: int = 0
    tool_invocations: int = 0
    condense_events: int = 0


@dataclass
class TraceMonitor:
    path: Path
    last_offset: int = 0

    def read_new_events(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        file_size = self.path.stat().st_size
        if file_size < self.last_offset:
            raise RuntimeError(
                f"Trace file shrank from {self.last_offset} to {file_size}; events not append-only."
            )
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.last_offset)
            data = handle.read()
            self.last_offset = handle.tell()
        lines = [line for line in data.splitlines() if line.strip()]
        events: list[dict[str, object]] = []
        for line in lines:
            events.append(json.loads(line))
        return events


def _notes_block(payload: dict[str, object]) -> str:
    return f"\n{NOTES_START_MARKER}\n{json.dumps(payload)}\n{NOTES_END_MARKER}\n"


def _tool_block(tool_calls: list[dict[str, object]]) -> str:
    return f"\n{TOOL_START_MARKER}\n{json.dumps(tool_calls)}\n{TOOL_END_MARKER}\n"


def _generate_notes_payload(rng: random.Random, turn_index: int) -> dict[str, object]:
    additions = rng.randint(1, 3)
    payload: dict[str, object] = {
        "add_open_loops": [f"loop-{turn_index}-{rng.randrange(1000)}" for _ in range(additions)],
        "add_memory_tags": [f"tag-{turn_index}-{rng.randrange(1000)}" for _ in range(additions)],
        "add_decisions": [f"decision-{turn_index}"],
        "add_constraints": [f"constraint-{turn_index}"],
    }
    if turn_index % 7 == 0:
        payload["set_goals"] = [f"goal-{turn_index}"]
    if turn_index % 11 == 0:
        payload["set_episode_summary"] = f"Summary for turn {turn_index}."
    return payload


def _enqueue_turn_responses(backend: FakeBackend, rng: random.Random, turn_index: int) -> None:
    reflection = f"Reflection {turn_index}: ok."
    planner = f"Plan {turn_index}: step {rng.randint(1, 3)}."
    critic = f"Critic {turn_index}: looks {rng.choice(['good', 'fine', 'safe'])}."

    backend.extend_role_responses("reflection", [reflection])
    backend.extend_role_responses("planner", [planner])
    backend.extend_role_responses("critic", [critic])

    notes_payload = _generate_notes_payload(rng, turn_index)
    notes_text = _notes_block(notes_payload)
    wants_tool = rng.random() < 0.35

    if wants_tool:
        tool_name = rng.choice(["fs.list_dir", "fs.read_text"])
        if tool_name == "fs.list_dir":
            tool_args = {"path": "."}
        else:
            target = rng.choice(["hello.txt", "notes.txt"])
            tool_args = {"path": target}
        tool_call_id = f"tool-{turn_index}-{rng.randrange(10000)}"
        tool_calls = [{"id": tool_call_id, "tool": tool_name, "args": tool_args}]
        initial = f"Inspecting sandbox for turn {turn_index}." + _tool_block(tool_calls)
        final = f"Turn {turn_index} complete." + notes_text
        backend.extend_role_responses("governor", [initial, final])
    else:
        governor = f"Turn {turn_index} complete." + notes_text
        backend.extend_role_responses("governor", [governor])


def _assert_state_invariants(state_limits: dict[str, int], episode_summary_max: int, state) -> None:
    memory_refs = state.memory_refs
    if len(set(memory_refs)) != len(memory_refs):
        raise RuntimeError("Duplicate IDs found in memory_refs.")
    granted = set(state.capabilities_granted)
    pending = set(state.capabilities_pending)
    if granted.intersection(pending):
        raise RuntimeError("Capabilities pending intersect with granted.")
    for field, limit in state_limits.items():
        value = getattr(state, field)
        if len(value) > limit:
            raise RuntimeError(f"State field {field} exceeds limit {limit} (len={len(value)}).")
    if len(state.episode_summary) > episode_summary_max:
        raise RuntimeError("Episode summary exceeds max length.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic soak test with FakeBackend.")
    parser.add_argument("--turns", type=int, default=100, help="Number of turns to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for responses.")
    parser.add_argument(
        "--session-id", type=str, default="soak-1", help="Session identifier for tracing."
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base_dir = Path("data") / "soak"
    sandbox_root = base_dir / "sandbox"
    trace_dir = base_dir / "traces"
    checkpoint_dir = base_dir / "checkpoints"

    sandbox_root.mkdir(parents=True, exist_ok=True)
    (sandbox_root / "hello.txt").write_text("hello", encoding="utf-8")
    (sandbox_root / "notes.txt").write_text("soak notes", encoding="utf-8")

    registry = ToolRegistry()
    registry.register("fs.list_dir", list_dir_handler(sandbox_root))
    registry.register("fs.read_text", read_text_handler(sandbox_root))

    executor = ToolExecutor(root=sandbox_root, registry=registry)
    tracer = TraceWriter(args.session_id, base_dir=trace_dir, run_id=str(args.seed))
    checkpoint = load_or_create(args.session_id, base_dir=checkpoint_dir)

    roles = [
        RoleSpec(name="reflection", system_prompt="Reflect briefly."),
        RoleSpec(name="planner", system_prompt="Plan the response."),
        RoleSpec(name="critic", system_prompt="Critique the plan."),
        RoleSpec(name="governor", system_prompt="Use tools and answer."),
    ]

    backend = FakeBackend()
    metrics = SoakMetrics()
    trace_monitor = TraceMonitor(path=tracer.path)

    policy = CondensePolicy()
    state_limits = {
        "goals": policy.max_goals,
        "open_loops": policy.max_open_loops,
        "decisions": policy.max_decisions,
        "constraints": policy.max_constraints,
        "memory_tags": policy.max_memory_tags,
        "memory_refs": MAX_MEMORY_REFS,
        "capabilities_granted": MAX_CAPABILITIES,
        "capabilities_pending": MAX_CAPABILITIES,
    }

    start_time = time.time()
    for turn_index in range(args.turns):
        _enqueue_turn_responses(backend, rng, turn_index)
        user_text = f"Soak turn {turn_index}"
        _final_text, _results, checkpoint = run_pipeline(
            checkpoint,
            user_text,
            roles,
            backend,
            tool_executor=executor,
            tracer=tracer,
        )
        checkpoint_path = save_checkpoint(checkpoint, base_dir=checkpoint_dir)

        metrics.turns += 1
        metrics.max_open_loops = max(metrics.max_open_loops, len(checkpoint.state.open_loops))
        metrics.max_memory_tags = max(metrics.max_memory_tags, len(checkpoint.state.memory_tags))

        new_events = trace_monitor.read_new_events()
        metrics.tool_invocations += sum(1 for event in new_events if event.get("kind") == "tool_start")
        metrics.condense_events += sum(1 for event in new_events if event.get("kind") == "condense")

        if tracer.path.exists():
            metrics.trace_file_size = tracer.path.stat().st_size
        metrics.checkpoint_size = checkpoint_path.stat().st_size

        _assert_state_invariants(state_limits, MAX_EPISODE_SUMMARY_CHARS, checkpoint.state)

    duration = time.time() - start_time
    print("Soak run complete.")
    print(f"Turns: {metrics.turns}")
    print(f"Trace file: {tracer.path} ({metrics.trace_file_size} bytes)")
    print(f"Checkpoint: {checkpoint_path} ({metrics.checkpoint_size} bytes)")
    print(f"Max open loops: {metrics.max_open_loops}")
    print(f"Max memory tags: {metrics.max_memory_tags}")
    print(f"Tool invocations: {metrics.tool_invocations}")
    print(f"Condense events: {metrics.condense_events}")
    print(f"Elapsed seconds: {duration:.2f}")


if __name__ == "__main__":
    main()
