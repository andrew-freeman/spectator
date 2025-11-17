# app/response/responder_hard_v3.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.core.schemas import GovernorDecision, PlannerPlan, ReflectionOutput


class ResponderHardV3:
    """
    Deterministic, minimal responder for Spectator V3.

    Design goals:
    - No persona or emotional colouring.
    - Short, factual responses.
    - Uses:
      - governor verdict as ground truth for safety,
      - tool results for observations,
      - planner final step for knowledge answers.
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
            reason = decision.rationale or "The plan was classified as unsafe."
            return f"I did not execute the requested action because it was considered unsafe. Reason: {reason}"

        if decision.verdict == "request_more_data":
            reason = decision.rationale or "More detail is required before proceeding."
            return f"I need more information before I can proceed. {reason}"

        # Normalize mode to reflection mode if empty
        mode = mode or reflection.mode

        # 2. Mode-specific handling
        if mode == "chat":
            return self._handle_chat_minimal(original_message)

        if mode == "knowledge":
            return self._handle_knowledge(plan, reflection)

        if mode == "world_query":
            return self._handle_world_query(tool_results)

        if mode == "world_control":
            return self._handle_world_control(tool_results)

        # 3. Fallback: use planner analysis or reflection goal
        if plan.analysis:
            return plan.analysis
        return reflection.goal or "No actionable response was produced."

    # ------------------------------------------------------------------
    # CHAT MODE (minimal)
    # ------------------------------------------------------------------
    def _handle_chat_minimal(self, user_message: str) -> str:
        text = (user_message or "").strip().lower()
        if "who are you" in text or "what are you" in text:
            return "I am a local reasoning process running on this machine."
        if "are you there" in text or "are you online" in text:
            return "Yes. I am active and ready for the next instruction."
        return "Message acknowledged. What would you like me to do?"

    # ------------------------------------------------------------------
    # KNOWLEDGE MODE
    # ------------------------------------------------------------------
    def _handle_knowledge(self, plan: PlannerPlan, reflection: ReflectionOutput) -> str:
        """
        Return the final reasoning step as the answer, if present.
        Otherwise fall back to the plan analysis or the original goal.
        """
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
                    summaries.append(f"GPU temperatures: {temps_str}.")
            elif tool == "read_gpu_memory":
                mem = payload.get("memory_used_mb")
                if isinstance(mem, list) and mem:
                    mem_str = ", ".join(f"{m} MB" for m in mem)
                    summaries.append(f"GPU memory usage: {mem_str}.")
            elif tool == "read_sensors":
                if payload:
                    summaries.append("Sensor readings were retrieved.")
            elif tool == "read_state":
                summaries.append("The latest shared state snapshot was retrieved.")
            elif tool == "query_memory":
                summaries.append("Relevant memories were retrieved.")
            elif tool == "run_system_command":
                summaries.append("A system command was executed and its output was captured.")

        if summaries:
            return " ".join(summaries)

        return "I attempted to read the requested information, but no useful data was returned."

    # ------------------------------------------------------------------
    # WORLD_CONTROL MODE
    # ------------------------------------------------------------------
    def _handle_world_control(self, tool_results: Sequence[Dict[str, Any]]) -> str:
        actions: List[str] = []

        for result in tool_results:
            tool = result.get("tool")
            status = result.get("status")
            payload = result.get("result") or {}

            if status != "ok":
                continue

            if tool == "set_fan_speed":
                speed = payload.get("fan_speed")
                if speed is not None:
                    actions.append(f"Fan speed set to {float(speed):.0f}%.")
                else:
                    actions.append("Fan speed update was requested.")
            elif tool == "update_state":
                actions.append("Shared state was updated.")
            elif tool == "append_memory":
                actions.append("A new memory entry was stored.")
            elif tool == "noop_control":
                reason = payload.get("reason") or "no-op control executed."
                actions.append(f"No-op control executed ({reason}).")

        if actions:
            return " ".join(actions)

        return "No control actions were applied in this cycle."


__all__ = ["ResponderHardV3"]
