# app/response/responder_soft_v3.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.core.schemas import GovernorDecision, PlannerPlan, ReflectionOutput


class ResponderSoftV3:
    """
    Friendlier, persona-aware responder for Spectator V3.

    - Uses identity profile for chat.
    - Uses planner final step for knowledge answers.
    - Provides slightly more natural phrasing for query/control modes.
    """

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
        # 1. Hard rejections / clarification
        if decision.verdict == "reject":
            base = (
                "I couldn't safely carry out that request because it would "
                "have violated policy or was classified as too risky."
            )
            extra = f" Details: {decision.rationale}" if decision.rationale else ""
            return base + extra

        if decision.verdict == "request_more_data":
            detail = decision.rationale or "I need more detail before I can proceed."
            return f"I need a bit more information before acting on that. {detail}"

        mode = mode or reflection.mode

        # 2. Mode-specific responses
        if mode == "chat":
            return self._handle_chat(original_message, identity)

        if mode == "knowledge":
            return self._handle_knowledge(plan, reflection)

        if mode == "world_query":
            return self._handle_world_query(tool_results)

        if mode == "world_control":
            return self._handle_world_control(tool_results, policy)

        # 3. Fallback
        if plan.analysis:
            return plan.analysis
        return reflection.goal or "I'm ready for your next instruction."

    # ------------------------------------------------------------------
    # CHAT MODE (soft, identity-aware)
    # ------------------------------------------------------------------
    def _handle_chat(self, user_message: str, identity: Dict[str, Any]) -> str:
        name = identity.get("name", "Spectator")
        desc = identity.get(
            "description",
            "a local autonomous reasoning agent running on this workstation",
        )

        # Trim leading "I am"/"I'm" in description if present
        lowered = desc.lower().strip()
        for prefix in ("i am ", "i'm ", "i’m "):
            if lowered.startswith(prefix):
                desc = desc[len(prefix):].lstrip()
                break

        if desc.endswith("."):
            desc = desc[:-1].rstrip()

        msg_lower = (user_message or "").strip().lower()

        if "who are you" in msg_lower or "what are you" in msg_lower:
            base = desc or "a local reasoning agent on this machine"
            return f"I'm {name}, {base}. I monitor your system, reason about your requests, and act within a safety policy."

        if "are you there" in msg_lower or "are you online" in msg_lower:
            base = desc or "your local reasoning agent"
            return f"I'm {name}, {base}, and I'm here. What would you like me to look at?"

        base = desc or "your local reasoning assistant on this workstation"
        return f"I'm {name}, {base}. How can I help you right now?"

    # ------------------------------------------------------------------
    # KNOWLEDGE MODE
    # ------------------------------------------------------------------
    def _handle_knowledge(self, plan: PlannerPlan, reflection: ReflectionOutput) -> str:
        if plan.steps:
            final_step = plan.steps[-1].strip()
            if final_step:
                return final_step

        if plan.analysis:
            return plan.analysis

        return reflection.goal

    # ------------------------------------------------------------------
    # WORLD_QUERY MODE
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
                if isinstance(temps, list) and temps:
                    temps_str = ", ".join(f"{t}°C" for t in temps)
                    summaries.append(f"The current GPU temperatures are {temps_str}.")
            elif tool == "read_gpu_memory":
                mem = payload.get("memory_used_mb")
                if isinstance(mem, list) and mem:
                    mem_str = ", ".join(f"{m} MB" for m in mem)
                    summaries.append(f"Current GPU memory usage: {mem_str}.")
            elif tool == "read_sensors":
                if payload:
                    summaries.append("I fetched the latest sensor readings.")
            elif tool == "read_state":
                summaries.append("I retrieved the latest controller state snapshot.")
            elif tool == "query_memory":
                summaries.append("I looked up the relevant stored memories.")
            elif tool == "run_system_command":
                summaries.append("I executed the requested system command and captured its output.")

        if summaries:
            return " ".join(summaries)

        return "I tried to read the system state, but I didn't receive any useful data."

    # ------------------------------------------------------------------
    # WORLD_CONTROL MODE
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
                if isinstance(temps, list) and temps:
                    temps_msg = "Current GPU temperatures: " + ", ".join(f"{t}°C" for t in temps) + "."

            if tool == "set_fan_speed" and status == "ok":
                speed = payload.get("fan_speed")
                reason = payload.get("reason")
                if speed is not None:
                    msg = f"I set the fan speed to {float(speed):.0f}% within the safety policy"
                    if reason:
                        msg += f" because {reason}."
                    else:
                        msg += "."
                    action_msgs.append(msg)

            if tool == "update_state" and status == "ok":
                action_msgs.append("I updated the shared controller state.")

            if tool == "append_memory" and status == "ok":
                action_msgs.append("I stored a new memory entry about this cycle.")

            if tool == "noop_control" and status == "ok":
                reason = payload.get("reason") or "no-op control executed."
                action_msgs.append(f"I didn't change any controls ({reason}).")

        policy_note = " I stayed within the configured safety and thermal policy."

        if temps_msg or action_msgs:
            return " ".join(filter(None, [temps_msg, *action_msgs])) + policy_note

        return (
            "I attempted to act on your request, but no control actions were actually "
            "performed. The policy or the current conditions may have prevented changes."
        )


__all__ = ["ResponderSoftV3"]