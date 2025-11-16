from typing import Any, Dict, List, Sequence

from app.core.schemas import GovernorDecision, PlannerPlan, ReflectionOutput, ResponderFrame


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
    ) -> ResponderFrame:
        verdict = decision.verdict
        used_tools = [result.get("tool") for result in tool_results if result.get("status") == "ok" and result.get("tool")]

        if verdict == "reject":
            text = (
                "I could not carry that out because it conflicted with the safety policy. "
                + (decision.rationale or "Please adjust the request and try again.")
            )
            return ResponderFrame(final_text=text, short_summary="Request rejected", used_tools=[], mode=reflection.mode)

        if verdict == "request_more_data":
            text = "I need more detail before proceeding. Could you clarify which readings or adjustments you expect?"
            return ResponderFrame(final_text=text, short_summary="Requested clarification", used_tools=[], mode=reflection.mode)

        mode = mode or reflection.mode

        if mode == "chat":
            final_text = self._handle_chat(original_message, identity)
        elif mode == "knowledge":
            final_text = self._handle_knowledge(plan, reflection)
        elif mode == "world_query":
            final_text = self._handle_world_query(tool_results)
        elif mode == "world_control":
            final_text = self._handle_world_control(tool_results, policy)
        else:
            final_text = plan.analysis or reflection.goal or "I'm ready for the next instruction."

        summary = self._summarize(plan, decision)
        cleaned_tools = [t for t in used_tools if isinstance(t, str)]
        return ResponderFrame(final_text=final_text, short_summary=summary, used_tools=cleaned_tools, mode=mode)

    def _handle_chat(self, user_message: str, identity: Dict[str, Any]) -> str:
        name = identity.get("name", "Spectator")
        role = identity.get("role", "a local assistant")
        environment = identity.get("environment", "this workstation")
        message = user_message.strip() or "your message"
        return f"{name} here—{role} operating on {environment}. Thanks for reaching out about {message}."

    def _handle_knowledge(self, plan: PlannerPlan, reflection: ReflectionOutput) -> str:
        if plan.steps:
            final_step = plan.steps[-1].strip()
            if final_step:
                return final_step
        if plan.analysis:
            return plan.analysis
        return reflection.goal

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
                    temps_str = " and ".join(f"{t}°C" for t in temps)
                    summaries.append(f"Current GPU temperatures are {temps_str}.")
            elif tool == "read_state":
                summaries.append("I retrieved the current controller state snapshot.")
            elif tool == "read_sensors":
                summaries.append("Sensor readings are cached for review.")
        if summaries:
            return " ".join(summaries)
        return "I attempted to gather system information but nothing useful was returned."

    def _handle_world_control(self, tool_results: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> str:
        action_msgs: List[str] = []
        for result in tool_results:
            tool = result.get("tool")
            status = result.get("status")
            payload = result.get("result") or {}
            if tool == "set_fan_speed" and status == "ok":
                speed = payload.get("fan_speed")
                reason = payload.get("reason") or "stabilizing temperatures"
                if speed is not None:
                    action_msgs.append(
                        f"Set the fan speed to {float(speed):.0f}% within the active policy because {reason}."
                    )
            if tool == "noop_control" and status == "ok":
                reason = payload.get("reason") or "policy limited changes"
                action_msgs.append(f"No control change was required ({reason}).")
        if action_msgs:
            return " ".join(action_msgs)
        policy_note = policy.get("thermal_policy", {}).get("description")
        if policy_note:
            return f"No adjustments were performed because {policy_note}."
        return "I did not apply any control changes because the policy prevented unsafe actions."

    def _summarize(self, plan: PlannerPlan, decision: GovernorDecision) -> str:
        if decision.verdict == "query_mode":
            return "Executed read-only query"
        if decision.verdict == "approve" and plan.tool_calls:
            return "Executed control plan"
        if plan.steps:
            return f"Planned steps: {len(plan.steps)}"
        return "Plan acknowledged"


__all__ = ["Responder"]
