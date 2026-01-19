from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Anomaly:
    code: str
    severity: str
    evidence: str
    category: str
    invariant: str


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            events.append(
                {
                    "ts": None,
                    "kind": "trace_parse_error",
                    "data": {"line": index, "raw": stripped[:200]},
                }
            )
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _load_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _bare_tool_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if "name" in payload and "arguments" in payload:
        return True
    if "tool" in payload and "args" in payload:
        return True
    if "tool" in payload and "arguments" in payload:
        return True
    return False


def _categorize_anomaly(code: str) -> tuple[str, str]:
    mapping = {
        "tool_calls_parse_warning": (
            "tool_call_format",
            "Tool calls must be canonical or parseable.",
        ),
        "visible_tool_json_leak": (
            "visible_leak",
            "Visible output must not contain tool-call payloads.",
        ),
        "tool_failed": (
            "tool_execution",
            "Tool execution must succeed or surface error explicitly.",
        ),
        "tool_missing_done": (
            "tool_execution",
            "Tool execution must produce a tool_done event.",
        ),
        "llm_req_done_mismatch": (
            "trace_integrity",
            "Trace must pair llm_req and llm_done events.",
        ),
        "sanitize_warning": (
            "sanitize_output",
            "Sanitizer must not empty visible output.",
        ),
        "tool_results_truncated": (
            "tool_results_budget",
            "Tool results should fit within the configured budget.",
        ),
        "trace_parse_error": (
            "trace_integrity",
            "Trace lines must be valid JSON.",
        ),
    }
    return mapping.get(code, ("unknown", "Unmapped invariant"))


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []
    for item in items:
        key = (item.get("action", ""), item.get("rationale", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def autopsy_from_trace(
    trace_path: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    events = _load_trace_events(trace_path)
    checkpoint = _load_checkpoint(checkpoint_path)

    stages: list[dict[str, Any]] = []
    open_stages: dict[str, list[dict[str, Any]]] = {}
    tool_entries: dict[str, dict[str, Any]] = {}
    tool_order: list[str] = []
    truncated_tools: set[str] = set()
    sanitizer_actions: list[dict[str, Any]] = []
    sanitizer_warnings: list[dict[str, Any]] = []
    anomalies: list[Anomaly] = []

    llm_req_count = 0
    llm_done_count = 0
    tool_start_ids: set[str] = set()
    tool_done_ids: set[str] = set()
    final_visible: str | None = None

    for event in events:
        kind = event.get("kind")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        role = data.get("role") if isinstance(data.get("role"), str) else None

        if kind == "trace_parse_error":
            category, invariant = _categorize_anomaly("trace_parse_error")
            anomalies.append(
                Anomaly(
                    code="trace_parse_error",
                    severity="warn",
                    evidence=f"line={data.get('line')}",
                    category=category,
                    invariant=invariant,
                )
            )
        if kind == "llm_req":
            llm_req_count += 1
            if role and role in open_stages and open_stages[role]:
                last_stage = open_stages[role][-1]
                if last_stage.get("llm_done_ts") is None:
                    category, invariant = _categorize_anomaly("llm_req_done_mismatch")
                    anomalies.append(
                        Anomaly(
                            code="llm_req_done_mismatch",
                            severity="warn",
                            evidence=f"role={role} missing llm_done before new llm_req",
                            category=category,
                            invariant=invariant,
                        )
                    )
            prompt = data.get("prompt")
            entry = {
                "role": role,
                "llm_req_ts": event.get("ts"),
                "llm_done_ts": None,
                "llm_req_chars": len(prompt) if isinstance(prompt, str) else None,
                "llm_done_chars": None,
            }
            if role:
                open_stages.setdefault(role, []).append(entry)
            stages.append(entry)
        elif kind == "llm_done":
            llm_done_count += 1
            response = data.get("response")
            entry = None
            if role and role in open_stages and open_stages[role]:
                entry = open_stages[role].pop()
            if entry is None:
                entry = {
                    "role": role,
                    "llm_req_ts": None,
                    "llm_done_ts": event.get("ts"),
                    "llm_req_chars": None,
                    "llm_done_chars": len(response) if isinstance(response, str) else None,
                }
                stages.append(entry)
            else:
                entry["llm_done_ts"] = event.get("ts")
                entry["llm_done_chars"] = (
                    len(response) if isinstance(response, str) else None
                )
        elif kind == "tool_start":
            tool_id = data.get("id")
            tool = data.get("tool")
            if isinstance(tool_id, str):
                tool_start_ids.add(tool_id)
                tool_entries.setdefault(
                    tool_id,
                    {
                        "id": tool_id,
                        "tool": tool,
                        "args": data.get("args"),
                        "duration_ms": None,
                        "ok": None,
                        "error": None,
                        "truncated": False,
                    },
                )
                if tool_id not in tool_order:
                    tool_order.append(tool_id)
        elif kind == "tool_done":
            tool_id = data.get("id")
            if isinstance(tool_id, str):
                tool_done_ids.add(tool_id)
                entry = tool_entries.setdefault(
                    tool_id,
                    {
                        "id": tool_id,
                        "tool": data.get("tool"),
                        "args": data.get("args"),
                        "duration_ms": None,
                        "ok": None,
                        "error": None,
                        "truncated": False,
                    },
                )
                entry["tool"] = data.get("tool", entry.get("tool"))
                entry["args"] = data.get("args", entry.get("args"))
                entry["duration_ms"] = data.get("duration_ms")
                entry["ok"] = data.get("ok")
                entry["error"] = data.get("error")
        elif kind == "tool_result_truncated":
            tools = data.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, str):
                        truncated_tools.add(tool)
        elif kind == "sanitize":
            sanitizer_actions.append(data)
        elif kind == "sanitize_warning":
            sanitizer_warnings.append(data)
            category, invariant = _categorize_anomaly("sanitize_warning")
            anomalies.append(
                Anomaly(
                    code="sanitize_warning",
                    severity="warn",
                    evidence=str(data.get("message", "sanitize_warning")),
                    category=category,
                    invariant=invariant,
                )
            )
        elif kind == "tool_calls_parse_warning":
            category, invariant = _categorize_anomaly("tool_calls_parse_warning")
            anomalies.append(
                Anomaly(
                    code="tool_calls_parse_warning",
                    severity="warn",
                    evidence=str(data.get("reason", "parse_warning")),
                    category=category,
                    invariant=invariant,
                )
            )
        elif kind == "visible_response":
            visible = data.get("visible_response")
            if isinstance(visible, str):
                final_visible = visible

    for entry in tool_entries.values():
        tool_name = entry.get("tool")
        if isinstance(tool_name, str) and tool_name in truncated_tools:
            entry["truncated"] = True

    for tool_id in tool_start_ids - tool_done_ids:
        category, invariant = _categorize_anomaly("tool_missing_done")
        anomalies.append(
            Anomaly(
                code="tool_missing_done",
                severity="high",
                evidence=f"id={tool_id}",
                category=category,
                invariant=invariant,
            )
        )

    for entry in tool_entries.values():
        if entry.get("ok") is False:
            category, invariant = _categorize_anomaly("tool_failed")
            anomalies.append(
                Anomaly(
                    code="tool_failed",
                    severity="high",
                    evidence=f"{entry.get('tool')}: {entry.get('error')}",
                    category=category,
                    invariant=invariant,
                )
            )

    if llm_req_count != llm_done_count:
        category, invariant = _categorize_anomaly("llm_req_done_mismatch")
        anomalies.append(
            Anomaly(
                code="llm_req_done_mismatch",
                severity="warn",
                evidence=f"llm_req={llm_req_count} llm_done={llm_done_count}",
                category=category,
                invariant=invariant,
            )
        )

    if final_visible and _bare_tool_json(final_visible):
        category, invariant = _categorize_anomaly("visible_tool_json_leak")
        anomalies.append(
            Anomaly(
                code="visible_tool_json_leak",
                severity="high",
                evidence=final_visible[:200],
                category=category,
                invariant=invariant,
            )
        )

    if any(kind == "tool_result_truncated" for kind in (e.get("kind") for e in events)):
        category, invariant = _categorize_anomaly("tool_results_truncated")
        anomalies.append(
            Anomaly(
                code="tool_results_truncated",
                severity="warn",
                evidence="tool_results_truncated",
                category=category,
                invariant=invariant,
            )
        )

    cause_categories: dict[str, dict[str, Any]] = {}
    for anomaly in anomalies:
        entry = cause_categories.setdefault(
            anomaly.category, {"invariant": anomaly.invariant, "evidence_codes": set()}
        )
        entry["evidence_codes"].add(anomaly.code)
    cause_summary = [
        {
            "category": category,
            "invariant": info["invariant"],
            "evidence_codes": sorted(info["evidence_codes"]),
        }
        for category, info in sorted(cause_categories.items())
    ]

    recommendations: list[dict[str, str]] = []
    for anomaly in anomalies:
        if anomaly.code == "visible_tool_json_leak":
            recommendations.append(
                {
                    "action": "Add or extend tool-call parsing tests for bare JSON leaks.",
                    "rationale": "Visible output contained a tool-call payload.",
                }
            )
        elif anomaly.code == "tool_calls_parse_warning":
            recommendations.append(
                {
                    "action": "Prefer canonical TOOL_CALLS_JSON wrapper in prompts.",
                    "rationale": "Tool-call parser emitted warnings.",
                }
            )
        elif anomaly.code == "tool_failed":
            recommendations.append(
                {
                    "action": "Verify tool args and allowlists for failing tool.",
                    "rationale": "Tool execution returned ok=false.",
                }
            )
        elif anomaly.code == "tool_missing_done":
            recommendations.append(
                {
                    "action": "Inspect tool executor for missing tool_done events.",
                    "rationale": "Tool started without completion.",
                }
            )
        elif anomaly.code == "sanitize_warning":
            recommendations.append(
                {
                    "action": "Review sanitizer rules for unexpected output removal.",
                    "rationale": "Sanitizer reported empty output.",
                }
            )
        elif anomaly.code == "tool_results_truncated":
            recommendations.append(
                {
                    "action": "Reduce tool output size or raise tool result budget.",
                    "rationale": "Tool results were truncated.",
                }
            )
        elif anomaly.code == "llm_req_done_mismatch":
            recommendations.append(
                {
                    "action": "Check trace logging around llm_req/llm_done.",
                    "rationale": "Trace has mismatched request/response events.",
                }
            )
        elif anomaly.code == "trace_parse_error":
            recommendations.append(
                {
                    "action": "Validate trace JSONL writer integrity.",
                    "rationale": "Trace contains invalid JSON lines.",
                }
            )

    tool_list = [tool_entries[tool_id] for tool_id in tool_order]

    summary: dict[str, Any] = {
        "trace_path": str(trace_path),
        "event_count": len(events),
        "roles": sorted(
            {
                event.get("data", {}).get("role")
                for event in events
                if isinstance(event.get("data"), dict)
                and isinstance(event.get("data", {}).get("role"), str)
            }
        ),
        "tool_count": len(tool_list),
        "anomaly_count": len(anomalies),
        "sanitizer_warning_count": len(sanitizer_warnings),
        "cause_categories": cause_summary,
    }
    if checkpoint:
        summary["checkpoint"] = {
            "session_id": checkpoint.get("session_id"),
            "revision": checkpoint.get("revision"),
            "updated_ts": checkpoint.get("updated_ts"),
            "trace_tail": checkpoint.get("trace_tail", []),
            "state_summary": {
                "goals": len(checkpoint.get("state", {}).get("goals", [])),
                "open_loops": len(checkpoint.get("state", {}).get("open_loops", [])),
                "decisions": len(checkpoint.get("state", {}).get("decisions", [])),
                "constraints": len(checkpoint.get("state", {}).get("constraints", [])),
            },
        }

    return {
        "summary": summary,
        "stages": stages,
        "tools": tool_list,
        "sanitizer": {
            "actions": sanitizer_actions,
            "warnings": sanitizer_warnings,
        },
        "anomalies": [asdict(anomaly) for anomaly in anomalies],
        "recommendations": _dedupe(recommendations),
    }


def render_autopsy_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    stages = report.get("stages", [])
    tools = report.get("tools", [])
    anomalies = report.get("anomalies", [])
    recommendations = report.get("recommendations", [])
    sanitizer = report.get("sanitizer", {})

    lines: list[str] = ["# Cognitive Autopsy Report", ""]
    lines.append("## Summary")
    lines.append(f"- Trace: `{summary.get('trace_path')}`")
    lines.append(f"- Events: {summary.get('event_count')}")
    lines.append(f"- Roles: {', '.join(summary.get('roles', [])) or 'none'}")
    lines.append(f"- Tools: {summary.get('tool_count')}")
    lines.append(f"- Anomalies: {summary.get('anomaly_count')}")
    lines.append(f"- Sanitizer warnings: {summary.get('sanitizer_warning_count')}")
    lines.append("")

    if summary.get("cause_categories"):
        lines.append("## Likely Causes")
        for entry in summary["cause_categories"]:
            codes = ", ".join(entry.get("evidence_codes", []))
            lines.append(
                f"- {entry.get('category')}: {entry.get('invariant')} (evidence: {codes})"
            )
        lines.append("")

    if stages:
        lines.append("## Stages")
        for stage in stages:
            role = stage.get("role") or "unknown"
            req_chars = stage.get("llm_req_chars")
            done_chars = stage.get("llm_done_chars")
            lines.append(f"- {role}: req_chars={req_chars} done_chars={done_chars}")
        lines.append("")

    if tools:
        lines.append("## Tools")
        for entry in tools:
            status = "ok" if entry.get("ok") else "error"
            truncated = " truncated" if entry.get("truncated") else ""
            lines.append(
                f"- {entry.get('tool')} id={entry.get('id')} status={status}{truncated} "
                f"duration_ms={entry.get('duration_ms')}"
            )
        lines.append("")

    if sanitizer.get("actions") or sanitizer.get("warnings"):
        lines.append("## Sanitizer")
        for action in sanitizer.get("actions", []):
            removed = action.get("removed")
            lines.append(f"- action: removed={removed}")
        for warning in sanitizer.get("warnings", []):
            lines.append(f"- warning: {warning.get('message')}")
        lines.append("")

    if anomalies:
        lines.append("## Anomalies")
        for anomaly in anomalies:
            lines.append(
                f"- {anomaly.get('severity')} {anomaly.get('code')}: {anomaly.get('evidence')}"
            )
        lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        for rec in recommendations:
            lines.append(f"- {rec.get('action')} ({rec.get('rationale')})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
