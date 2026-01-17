from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectator.runtime.condense import CondensePolicy

MAX_EPISODE_SUMMARY_CHARS = 2000
DEFAULT_TRACE_BYTES_PER_TURN_WARN = 50_000
DEFAULT_CONDENSE_STATE_PER_TURN_WARN = 0.5
DEFAULT_NOTES_PATCH_PER_TURN_WARN = 1.0

DEFAULT_MAX_CONDENSE_STATE_DELTA = 0.2
DEFAULT_MAX_TOOL_FAIL_RATE_DELTA = 0.01
DEFAULT_MAX_TRACE_BYTES_PER_TURN_DELTA = 10_000


@dataclass(slots=True)
class AnalysisSummary:
    turns: int
    trace_bytes: int
    checkpoint_bytes: int
    trace_bytes_per_turn: float
    event_counts: dict[str, int]
    tool_counts: dict[str, int]
    tool_ok: int
    tool_fail: int
    condense_counts: dict[str, int]
    notes_patch_count: int
    condense_state_per_turn: float
    tool_fail_rate: float
    warnings: list[str]
    failures: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "trace_bytes": self.trace_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "trace_bytes_per_turn": self.trace_bytes_per_turn,
            "event_counts": self.event_counts,
            "tool_counts": self.tool_counts,
            "tool_ok": self.tool_ok,
            "tool_fail": self.tool_fail,
            "condense_counts": self.condense_counts,
            "notes_patch_count": self.notes_patch_count,
            "condense_state_per_turn": self.condense_state_per_turn,
            "tool_fail_rate": self.tool_fail_rate,
            "warnings": self.warnings,
            "failures": self.failures,
        }


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {index}.") from exc
    return events


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "state" not in payload:
        raise ValueError("Checkpoint missing state payload.")
    return payload


def _require_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Checkpoint state.{key} must be a list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint state.{key} must contain strings only.")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"Checkpoint state.{key} must be a string.")
    return value


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    policy: CondensePolicy,
    failures: list[str],
) -> None:
    state = checkpoint.get("state", {})
    memory_refs = _require_list(state, "memory_refs")
    if len(set(memory_refs)) != len(memory_refs):
        failures.append("Duplicate IDs found in memory_refs.")
    granted = set(_require_list(state, "capabilities_granted"))
    pending = set(_require_list(state, "capabilities_pending"))
    if granted.intersection(pending):
        failures.append("Capabilities pending intersect with granted.")

    state_limits = {
        "goals": policy.max_goals,
        "open_loops": policy.max_open_loops,
        "decisions": policy.max_decisions,
        "constraints": policy.max_constraints,
        "memory_tags": policy.max_memory_tags,
    }
    for field, limit in state_limits.items():
        items = _require_list(state, field)
        if len(items) > limit:
            failures.append(
                f"State field {field} exceeds limit {limit} (len={len(items)})."
            )

    episode_summary = _require_str(state, "episode_summary")
    if len(episode_summary) > MAX_EPISODE_SUMMARY_CHARS:
        failures.append("Episode summary exceeds max length.")


def _compare_baseline(
    summary: AnalysisSummary,
    baseline_path: Path,
    failures: list[str],
) -> list[str]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    for key in ("condense_state_per_turn", "tool_fail_rate", "trace_bytes_per_turn"):
        if key not in baseline:
            warnings.append(f"Baseline missing {key}; skipping.")
    if "condense_state_per_turn" in baseline:
        delta = summary.condense_state_per_turn - baseline["condense_state_per_turn"]
        if delta > DEFAULT_MAX_CONDENSE_STATE_DELTA:
            failures.append(
                f"condense_state_per_turn delta {delta:.3f} exceeds {DEFAULT_MAX_CONDENSE_STATE_DELTA:.3f}"
            )
    if "tool_fail_rate" in baseline:
        delta = summary.tool_fail_rate - baseline["tool_fail_rate"]
        if delta > DEFAULT_MAX_TOOL_FAIL_RATE_DELTA:
            failures.append(
                f"tool_fail_rate delta {delta:.3f} exceeds {DEFAULT_MAX_TOOL_FAIL_RATE_DELTA:.3f}"
            )
    if "trace_bytes_per_turn" in baseline:
        delta = summary.trace_bytes_per_turn - baseline["trace_bytes_per_turn"]
        if delta > DEFAULT_MAX_TRACE_BYTES_PER_TURN_DELTA:
            failures.append(
                "trace_bytes_per_turn delta "
                f"{delta:.0f} exceeds {DEFAULT_MAX_TRACE_BYTES_PER_TURN_DELTA:.0f}"
            )
    return warnings


def analyze_soak(
    trace_path: Path,
    checkpoint_path: Path,
    turns: int | None = None,
    baseline_path: Path | None = None,
    max_tool_fail_rate: float = 0.0,
) -> AnalysisSummary:
    failures: list[str] = []
    warnings: list[str] = []

    events = _load_trace_events(trace_path)
    if not events:
        failures.append("Trace contains no events.")

    event_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    condense_counts: dict[str, int] = {}
    tool_ok = 0
    tool_fail = 0
    tool_plan = 0
    tool_start = 0
    tool_done = 0
    llm_req = 0
    llm_done = 0
    notes_patch_count = 0
    tool_start_ids: set[str] = set()
    tool_done_ids: set[str] = set()

    for event in events:
        kind = event.get("kind")
        if not isinstance(kind, str):
            continue
        event_counts[kind] = event_counts.get(kind, 0) + 1
        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {}
        if kind == "llm_req":
            llm_req += 1
        elif kind == "llm_done":
            llm_done += 1
        elif kind == "tool_plan":
            tool_plan += 1
        elif kind == "tool_start":
            tool_start += 1
            tool_id = data.get("id")
            if isinstance(tool_id, str):
                tool_start_ids.add(tool_id)
            tool_name = data.get("tool")
            if isinstance(tool_name, str) and tool_name not in tool_counts:
                tool_counts[tool_name] = 0
        elif kind == "tool_done":
            tool_done += 1
            tool_id = data.get("id")
            if isinstance(tool_id, str):
                tool_done_ids.add(tool_id)
            tool_name = data.get("tool")
            if isinstance(tool_name, str):
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            ok_value = data.get("ok")
            if ok_value is True:
                tool_ok += 1
            elif ok_value is False:
                tool_fail += 1
        elif kind == "condense":
            scope = data.get("scope")
            if isinstance(scope, str):
                condense_counts[scope] = condense_counts.get(scope, 0) + 1
        elif kind == "notes_patch":
            notes_patch_count += 1

    if llm_req != llm_done:
        failures.append(f"llm_req ({llm_req}) != llm_done ({llm_done}).")
    if tool_plan != tool_start or tool_start != tool_done:
        failures.append(
            f"tool_plan ({tool_plan}), tool_start ({tool_start}), tool_done ({tool_done}) mismatch."
        )
    missing_done = tool_start_ids - tool_done_ids
    if missing_done:
        failures.append(
            f"tool_start missing tool_done for ids: {', '.join(sorted(missing_done))}."
        )
    tool_fail_rate = tool_fail / max(tool_done, 1)
    if tool_fail_rate > max_tool_fail_rate:
        failures.append(
            f"tool_fail_rate {tool_fail_rate:.3f} exceeds {max_tool_fail_rate:.3f}."
        )

    policy = CondensePolicy()
    checkpoint = _load_checkpoint(checkpoint_path)
    _validate_checkpoint(checkpoint, policy, failures)

    resolved_turns = turns or event_counts.get("notes_patch", 0)
    if resolved_turns <= 0:
        failures.append("Unable to infer turns from trace; pass --turns.")
        resolved_turns = max(turns or 0, 1)

    trace_bytes = trace_path.stat().st_size
    checkpoint_bytes = checkpoint_path.stat().st_size
    trace_bytes_per_turn = trace_bytes / resolved_turns
    condense_state = condense_counts.get("state", 0)
    condense_state_per_turn = condense_state / resolved_turns

    if condense_state_per_turn >= DEFAULT_CONDENSE_STATE_PER_TURN_WARN:
        warnings.append(
            f"High condense_state_per_turn ({condense_state_per_turn:.2f})."
        )
    if trace_bytes_per_turn > DEFAULT_TRACE_BYTES_PER_TURN_WARN:
        warnings.append(
            f"Trace bytes per turn high ({trace_bytes_per_turn:.0f})."
        )
    if notes_patch_count != resolved_turns:
        warnings.append(
            f"notes_patch count ({notes_patch_count}) != turns ({resolved_turns})."
        )

    summary = AnalysisSummary(
        turns=resolved_turns,
        trace_bytes=trace_bytes,
        checkpoint_bytes=checkpoint_bytes,
        trace_bytes_per_turn=trace_bytes_per_turn,
        event_counts=event_counts,
        tool_counts=tool_counts,
        tool_ok=tool_ok,
        tool_fail=tool_fail,
        condense_counts=condense_counts,
        notes_patch_count=notes_patch_count,
        condense_state_per_turn=condense_state_per_turn,
        tool_fail_rate=tool_fail_rate,
        warnings=warnings,
        failures=failures,
    )

    if baseline_path is not None:
        baseline_warnings = _compare_baseline(summary, baseline_path, failures)
        summary.warnings.extend(baseline_warnings)

    return summary


def render_summary(summary: AnalysisSummary) -> str:
    lines = [
        "Soak analysis summary",
        f"Turns: {summary.turns}",
        f"Trace bytes: {summary.trace_bytes} ({summary.trace_bytes_per_turn:.0f}/turn)",
        f"Checkpoint bytes: {summary.checkpoint_bytes}",
        f"Events: {summary.event_counts}",
        f"Tools: {summary.tool_counts} (ok={summary.tool_ok}, fail={summary.tool_fail})",
        f"Condense: {summary.condense_counts}",
        f"notes_patch: {summary.notes_patch_count}",
    ]
    if summary.warnings:
        lines.append(f"Warnings: {len(summary.warnings)}")
        lines.extend(f"- {warning}" for warning in summary.warnings[:5])
    if summary.failures:
        lines.append(f"Failures: {len(summary.failures)}")
        lines.extend(f"- {failure}" for failure in summary.failures[:5])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a soak run trace/checkpoint.")
    parser.add_argument("--trace", type=Path, required=True, help="Trace JSONL path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint JSON path.")
    parser.add_argument("--turns", type=int, default=None, help="Turn count (optional).")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline summary JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Optional summary JSON output.")
    parser.add_argument(
        "--max-tool-fail-rate",
        type=float,
        default=0.0,
        help="Allow tool failures up to this rate.",
    )
    parser.add_argument(
        "--fail-on-warn", action="store_true", help="Treat warnings as failures."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze_soak(
        trace_path=args.trace,
        checkpoint_path=args.checkpoint,
        turns=args.turns,
        baseline_path=args.baseline,
        max_tool_fail_rate=args.max_tool_fail_rate,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary.to_json(), indent=2), encoding="utf-8")

    print(render_summary(summary))

    exit_code = 0
    if summary.failures:
        exit_code = 2
    elif summary.warnings and args.fail_on_warn:
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
