"""Convert internal structured outputs into a conversational response."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)


def build_response(
    *,
    user_message: str,
    reflection_output: Optional[Dict[str, Any]],
    actor_output: Optional[Dict[str, Any]],
    critic_output: Optional[Dict[str, Any]],
    governor_decision: Optional[Dict[str, Any]],
    tool_results: Optional[List[Dict[str, Any]]],
    identity_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a safe, natural-language response for the user."""

    identity_profile = identity_profile or {}
    reflection_output = reflection_output or {}
    actor_output = actor_output or {}
    critic_output = critic_output or {}
    governor_decision = governor_decision or {}
    tool_results = tool_results or []

    try:
        if _is_identity_question(user_message):
            return _identity_description(identity_profile)

        intent = reflection_output.get("intent", "").lower()
        context = reflection_output.get("context") or {}

        if intent == "objective" or context.get("goal_update"):
            objectives = reflection_output.get("refined_objectives") or []
            return _objective_acknowledgement(objectives)

        if context.get("chat_mode"):
            return _chat_reply(user_message, identity_profile)

        verdict = str(governor_decision.get("verdict", "")).lower()

        if verdict == "query_mode":
            query_text = _format_query_mode_response(tool_results)
            if query_text:
                return query_text

        action_text = _format_action_confirmation(tool_results)
        if action_text:
            return action_text

        escalation_text = _format_escalation_message(verdict, governor_decision, critic_output)
        if escalation_text:
            return escalation_text

        actor_summary = _summarize_actor(actor_output)
        observation_text = _format_tool_observations(tool_results)
        if actor_summary and observation_text:
            return f"{actor_summary}\n\n{observation_text}"
        if actor_summary:
            return actor_summary
        if observation_text:
            return observation_text

        return _fallback_message(identity_profile)
    except Exception:  # pragma: no cover - defensive safety net
        LOGGER.exception("Failed to build response")
        return "I encountered an internal error while composing my reply. Please try again soon."


def _chat_reply(user_message: str, identity_profile: Dict[str, Any]) -> str:
    name = identity_profile.get("name", "Spectator")
    role = identity_profile.get("role", "an autonomous assistant")
    environment = identity_profile.get("environment", "this workstation")
    acknowledgement = user_message.strip() or "your message"
    return (
        f"{name} here—{role} operating within {environment}. "
        f"Thanks for reaching out; here's my take on what you said: {acknowledgement}."
    )


def _format_query_mode_response(tool_results: List[Dict[str, Any]]) -> str:
    observations = _format_tool_observations(tool_results)
    if not observations:
        return "I attempted to look up the information but nothing useful was returned."
    return f"Here is what I found based on the requested readings:\n{observations}"


def _format_action_confirmation(tool_results: List[Dict[str, Any]]) -> str:
    for result in tool_results:
        if result.get("status") != "ok":
            continue
        tool_name = result.get("tool")
        payload = result.get("result") or {}
        if tool_name == "set_fan_speed":
            speed = payload.get("fan_speed")
            reason = payload.get("reason") or "stabilizing temperatures"
            if speed is not None:
                return f"Fan speed set to {speed}% (reason: {reason})."
        if tool_name in {"update_state", "append_memory"}:
            return f"{tool_name.replace('_', ' ').title()} completed successfully."
    return ""


def _format_escalation_message(
    verdict: str,
    governor_decision: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> str:
    rationale = governor_decision.get("rationale") or ""
    critic_evaluation = critic_output.get("evaluation") or ""
    if verdict == "reject_plan":
        explanation = rationale or critic_evaluation or "The plan conflicted with current safeguards."
        return f"I rejected the proposed plan: {explanation}."
    if verdict in {"request_more_data", "defer_to_critic"}:
        explanation = rationale or critic_evaluation or "I need more information before proceeding."
        return f"I paused execution because: {explanation}."
    return ""


def _summarize_actor(actor_output: Dict[str, Any]) -> str:
    if not actor_output:
        return ""
    analysis = str(actor_output.get("analysis", "")).strip()
    steps = actor_output.get("plan") or actor_output.get("steps") or []
    if isinstance(steps, str):
        steps = [steps]
    info_gaps = actor_output.get("information_gaps") or []
    confidence = actor_output.get("confidence")

    parts: List[str] = []
    if analysis:
        parts.append(f"Analysis: {analysis}.")
    if isinstance(steps, list):
        plan_text = "; ".join(str(step).strip() for step in steps if str(step).strip())
        if plan_text:
            parts.append(f"Planned steps: {plan_text}.")
    if isinstance(info_gaps, list) and info_gaps:
        gaps = ", ".join(str(gap).strip() for gap in info_gaps if str(gap).strip())
        if gaps:
            parts.append(f"Information gaps: {gaps}.")
    try:
        if confidence is not None:
            confidence_pct = max(0.0, min(float(confidence) * 100.0, 100.0))
            parts.append(f"Confidence: {confidence_pct:.0f}%.")
    except (TypeError, ValueError):
        pass
    return " ".join(parts)


def _format_tool_observations(tool_results: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for result in tool_results:
        tool_name = result.get("tool") or result.get("name") or "tool"
        status = result.get("status") or "unknown"
        summary = _summarize_result_payload(result.get("result"))
        lines.append(f"- {tool_name}: {summary} (status: {status})")
    return "\n".join(lines)


def _summarize_result_payload(payload: Any) -> str:
    if payload is None:
        return "no additional details provided"
    if isinstance(payload, (str, int, float)):
        return str(payload)
    if isinstance(payload, list):
        if not payload:
            return "no entries"
        preview = ", ".join(str(item) for item in payload[:3])
        if len(payload) > 3:
            preview += ", …"
        return f"{len(payload)} entries ({preview})"
    if isinstance(payload, dict):
        items = []
        for idx, (key, value) in enumerate(payload.items()):
            if idx >= 4:
                items.append("…")
                break
            items.append(f"{key}: { _format_simple_value(value) }")
        return "; ".join(items) or "details captured"
    return str(payload)


def _format_simple_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value}"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "none"
        preview = ", ".join(str(item) for item in value[:2])
        if len(value) > 2:
            preview += ", …"
        return f"[{preview}]"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(keys[:3])
        if len(keys) > 3:
            preview += ", …"
        return f"details: {preview or 'n/a'}"
    return str(value)


def _objective_acknowledgement(objectives: List[str]) -> str:
    if not objectives:
        return "Objective noted. I'll keep it in memory for future cycles."
    if len(objectives) == 1:
        return f"Objective noted: {objectives[0]}. I'll keep it in memory for upcoming work."
    joined = "; ".join(objectives)
    return f"Recorded the following objectives: {joined}."


def _identity_description(identity_profile: Dict[str, Any]) -> str:
    description = identity_profile.get("description")
    if description:
        return description
    name = identity_profile.get("name", "Spectator")
    role = identity_profile.get("role", "an autonomous agent")
    environment = identity_profile.get("environment", "this machine")
    capabilities = identity_profile.get("capabilities") or []
    capability_text = ""
    if capabilities:
        preview = ", ".join(capabilities[:3])
        if len(capabilities) > 3:
            preview += ", and more"
        capability_text = f" I can {preview}."
    return f"I am {name}, {role} operating inside {environment}.{capability_text}"


def _fallback_message(identity_profile: Dict[str, Any]) -> str:
    name = identity_profile.get("name", "Spectator")
    return f"{name} is thinking, but I need more details to proceed. Could you clarify your request?"


def _is_identity_question(message: str) -> bool:
    lowered = (message or "").strip().lower()
    return "who are you" in lowered or lowered == "who r u"


__all__ = ["build_response"]
