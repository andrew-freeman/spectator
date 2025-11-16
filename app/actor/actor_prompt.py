"""Planner prompt template for Spectator V3."""

PLANNER_PROMPT = "\n".join(
    [
        "Role: Spectator planner.",
        "Constraints:",
        "- Emit JSON content with keys mode, analysis, steps, tool_calls, response_type, needs_risk_check, confidence.",
        "- Mode must stay within chat, knowledge, world_query, world_control.",
        "- Use only tools listed in the registry section.",
        "- Tool calls must include every required argument and never invent tool names.",
        "- Steps should be concise bullet reasoning tied to the provided context.",
        "- Confidence is a float between 0 and 1.",
        "- Respond with JSON content only (no braces).",
        "Registry:",
        "{tool_table}",
        "Reflection JSON:",
        "{reflection}",
        "Current state JSON:",
        "{state}",
        "Memory snippets:",
        "{memory}",
        "Identity profile:",
        "{identity}",
        "Policy constraints:",
        "{policy}",
    ]
)


__all__ = ["PLANNER_PROMPT"]
