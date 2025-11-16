"""Deterministic responder for Spectator."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.core.schemas import GovernorDecision, PlannerPlan, ReflectionOutput


class Responder:
    """Translate pipeline artifacts into natural language replies."""

    def build(
        self,
        *,
        mode: str,
        reflection: ReflectionOutput,
        plan: PlannerPlan,
        decision: GovernorDecision,
        tool_results: Sequence[Dict[str, Any]],
        identity: Dict[str, Any],
        policy: Dict[str, Any],
        original_message: str,
        current_state: Dict[str, Any],
    ) -> str:
        if decision.verdict == "reject":
            return (
                "I couldn't safely carry out that request because it would violate "
                "policy or posed too much risk. "
                + (decision.rationale or "Please adjust the request and try again.")
            )

        if decision.verdict == "request_more_data":
            return (
                "I need a bit more detail before acting on that. Could you clarify "
                "what information I should gather or which component to adjust?"
            )

        mode = mode or reflection.mode
        if mode == "chat":
            return self._handle_chat(original_message, identity)
        if mode == "knowledge":
            return self._handle_knowledge(plan, reflection)
        if mode == "world_query":
            return self._handle_world_query(tool_results)
        if mode == "world_control":
            return self._handle_world_control(tool_results, policy)
        return plan.analysis or reflection.goal or "I'm ready for the next instruction."

    def _handle_chat(self, user_message: str, identity: Dict[str, Any]) -> str:
        name = identity.get("name", "Spectator")
        description = identity.get(
            "description",
            "a local autonomous reasoning agent monitoring your workstation",
        )
        lower = user_message.lower()
        if "who are you" in lower or "what are you" in lower:
            return (
                f"I'm {name}, {description}. I monitor GPUs, adjust fans within the thermal"
                " policy, and keep all reasoning local on this machine."
            )
        if "are you there" in lower or "are you online" in lower:
            return (
                f"I'm {name}, online and monitoring your system. Let me know how I can help."
            )
        return (
            f"I'm {name}, your local reasoning agent. {description}."
            " How can I assist you further?"
        )

    def _handle_knowledge(self, plan: PlannerPlan, reflection: ReflectionOutput) -> str:
        if plan.analysis:
            return plan.analysis
        if plan.steps:
            return plan.steps[-1]
        return reflection.goal

    def _handle_world_query(self, tool_results: Sequence[Dict[str, Any]]) -> str:
        summaries = []
        for result in tool_results:
            if result.get("status") != "ok":
                continue
            tool = result.get("tool")
            payload = result.get("result") or {}
            if tool == "read_gpu_temps":
                temps = payload.get("gpu_temps")
                if temps:
                    temps_str = " and ".join(f"{t}°C" for t in temps)
                    summaries.append(f"The current GPU temperatures are {temps_str}.")
            elif tool == "read_state":
                summaries.append("I retrieved the latest state snapshot from the controller.")
            elif tool == "read_sensors":
                summaries.append("I fetched the cached sensor readings.")
        if summaries:
            return " ".join(summaries)
        return "I tried to read the system state, but didn't receive any useful data."

    def _handle_world_control(
        self,
        tool_results: Sequence[Dict[str, Any]],
        policy: Dict[str, Any],
    ) -> str:
        temps_msg = None
        action_msgs: List[str] = []
        for result in tool_results:
            tool = result.get("tool")
            status = result.get("status")
            payload = result.get("result") or {}
            if tool == "read_gpu_temps" and status == "ok":
                temps = payload.get("gpu_temps")
                if temps:
                    temps_msg = "Current GPU temperatures: " + ", ".join(f"{t}°C" for t in temps) + "."
            if tool == "set_fan_speed" and status == "ok":
                speed = payload.get("fan_speed")
                reason = payload.get("reason")
                if speed is not None:
                    action_msgs.append(
                        f"Set the fan speed to {speed:.0f}% within the safety policy"
                        + (f" because {reason}." if reason else ".")
                    )
        policy_note = " I remained within the thermal policy constraints."
        if temps_msg or action_msgs:
            return " ".join(filter(None, [temps_msg, *action_msgs])) + policy_note
        return (
            "I attempted to act on your request, but no control actions were performed."
            " The policy may have prevented unsafe changes."
        )


__all__ = ["Responder"]
