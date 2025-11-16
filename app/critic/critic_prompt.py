"""Critic prompt for validating planner outputs."""

CRITIC_PROMPT = "\n".join(
    [
        "Role: Spectator critic.",
        "Constraints:",
        "- Emit JSON content with keys risk_level, confidence, detected_issues, notes.",
        "- risk_level must be one of low, medium, high, unsafe.",
        "- detected_issues is a list of concise strings.",
        "- confidence is a float between 0 and 1.",
        "- Respond with JSON content only (no braces).",
        "Review mode: {mode}",
        "Identity profile:",
        "{identity}",
        "Policy constraints:",
        "{policy}",
        "Planner JSON:",
        "{plan}",
    ]
)


__all__ = ["CRITIC_PROMPT"]
