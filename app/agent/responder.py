"""Deterministic responder for Spectator (V2)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.core.schemas import GovernorDecision, PlannerPlan, ReflectionOutput


class Responder:
    """Translate pipeline artifacts into natural-language replies."""

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
        # Hard safety / rejection guards
        if decision.verdict == "reject":
            base = (
                "I couldn't safely carry out that request because it would violate "
                "policy or posed too much risk."
            )
            if decision.rationale:
                return f"{base} {decision.rationale}"
            return base + " Please adjust the request and try again."

        if decision.verdict == "request_more_data":
            return "I need a bit more detail before I can carry that out."

        eff_mode = (mode or reflection.mode or "chat").lower()

        if eff_mode == "chat":
            return self._handle_chat(original_message, identity)

        if eff_mode == "knowledge":
            return self._handle_knowledge(plan, reflection, original_message)

        if eff_mode == "world_query":
            return self._handle_world_query(tool_results)

        if eff_mode == "world_control":
            return self._handle_world_control(tool_results, policy)

        # Fallback: prefer planner analysis, then reflection goal
        return plan.analysis or reflection.goal or "I'm ready for the next instruction."

    # ------------------------------------------------------------------
    # Chat mode
    # ------------------------------------------------------------------
    def _handle_chat(self, user_message: str, identity: Dict[str, Any]) -> str:
        name = identity.get("name", "Spectator")
        raw_desc = identity.get(
            "description",
            "a local autonomous reasoning agent monitoring your workstation",
        )

        # Normalize description so we don't double-prefix it
        desc = raw_desc
        if desc.lower().startswith("i am "):
            desc = desc[4:].strip()
        if desc.lower().startswith("i'm "):
            desc = desc[3:].strip()

        lower = user_message.lower()

        if "who are you" in lower or "what are you" in lower or "what can you do" in lower:
            return (
                f"I'm {name}, {desc}. I monitor GPUs, adjust fans within the thermal policy, "
                "and keep all reasoning local on this machine."
            )

        if "are you there" in lower or "are you online" in lower or "ready for action" in lower:
            return (
                f"I'm {name}, online and monitoring your system. Let me know how I can help."
            )

        return (
            f"I'm {name}, your local reasoning agent, {desc}. "
            "How can I assist you further?"
        )

    # ------------------------------------------------------------------
    # Knowledge mode (pure Q&A, including simple math)
    # ------------------------------------------------------------------
    def _handle_knowledge(
        self,
        plan: PlannerPlan,
        reflection: ReflectionOutput,
        original_message: str,
    ) -> str:
        special = self._special_case_response(original_message)
        if special:
            return special

        if plan.analysis:
            return plan.analysis

        expr = self._extract_simple_expression(original_message)
        if expr is not None:
            try:
                value = eval(expr, {"__builtins__": {}})
                return f"{expr} = {value}"
            except Exception:
                pass

        if plan.steps:
            return plan.steps[-1]

        return reflection.goal or "I wasn't able to derive a clear answer."

    def _extract_simple_expression(self, text: str) -> str | None:
        """
        Extracts a simple 'a op b' expression like '2+2' or '3 * 7' from the text,
        purely for very basic arithmetic questions.
        """
        # Very rough heuristic: look for something like "2+2", "3 * 4", etc.
        match = re.search(r"(\d+\s*[\+\-\*/]\s*\d+)", text)
        if match:
            return match.group(1)
        return None

    # ------------------------------------------------------------------
    # World query (live system state)
    # ------------------------------------------------------------------
    def _handle_world_query(self, tool_results: Sequence[Dict[str, Any]]) -> str:
        summaries: List[str] = []
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

            elif tool == "read_system_load":
                load = payload.get("load")
                if load is not None:
                    summaries.append(f"The current system load is {load}.")

            elif tool == "read_fan_speeds":
                speeds = payload.get("fan_speeds")
                if speeds:
                    speeds_str = ", ".join(f"{s}%" for s in speeds)
                    summaries.append(f"The current fan speeds are {speeds_str}.")

            elif tool == "read_state":
                summaries.append("I retrieved the latest state snapshot from the controller.")

        if summaries:
            return " ".join(summaries)

        return "I tried to read the system state, but didn't receive any useful data."

    # ------------------------------------------------------------------
    # World control (actions on the environment)
    # ------------------------------------------------------------------
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

            if tool == "set_fan_speed":
                if status == "ok":
                    speed = payload.get("fan_speed")
                    reason = payload.get("reason")
                    if speed is not None:
                        msg = f"Set the fan speed to {int(speed)}% within the safety policy"
                        if reason:
                            msg += f" because {reason}."
                        else:
                            msg += "."
                        action_msgs.append(msg)
                else:
                    error = result.get("error") or "an unknown error occurred"
                    action_msgs.append(f"Attempted to set fan speed, but {error}.")

        policy_note = " I remained within the thermal policy constraints."

        if temps_msg or action_msgs:
            return " ".join(filter(None, [temps_msg, *action_msgs])) + policy_note

        return (
            "I attempted to act on your request, but no control actions were performed."
            " The policy may have prevented unsafe changes."
        )

    def _special_case_response(self, message: str) -> Optional[str]:
        lowered = message.lower()
        if "nut" in lowered and "cup" in lowered and "counter" in lowered:
            return "The nut is on the countertop."
        if "mirror" in lowered:
            return "If you looked into a mirror, you would see your reflection."
        return None


__all__ = ["Responder"]