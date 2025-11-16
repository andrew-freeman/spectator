"""Decision logic for the Spectator governor layer."""
from __future__ import annotations

from app.core.schemas import CriticOutput, GovernorDecision, Plan, PreprocessorOutput


def decide(
    preproc: PreprocessorOutput,
    plan: Plan,
    critic: CriticOutput,
) -> GovernorDecision:
    """Return a governor verdict for the provided cycle artifacts."""

    if preproc.needs_clarification and preproc.clarification_question:
        return GovernorDecision(
            verdict="reject",
            final_response_mode="text",
            ask_user=preproc.clarification_question,
            notes="Clarification requested by preprocessor.",
        )

    if critic.risk in {"high", "unsafe"}:
        issues = "; ".join(critic.issues) if critic.issues else critic.notes
        return GovernorDecision(
            verdict="reject",
            final_response_mode="text",
            notes="Plan rejected due to high/unsafe risk: " + (issues or "No details."),
        )

    if preproc.mode == "chat" and not preproc.requires_tools:
        return GovernorDecision(
            verdict="chat_only",
            final_steps=plan.steps,
            final_tool_calls=[],
            final_response_mode=plan.response_type,
            notes="Chat-only mode.",
        )

    if preproc.mode == "knowledge" and not preproc.requires_tools:
        return GovernorDecision(
            verdict="chat_only",
            final_steps=plan.steps,
            final_tool_calls=[],
            final_response_mode=plan.response_type,
            notes="Knowledge-only answer.",
        )

    final_steps = critic.adjusted_steps or plan.steps
    final_tool_calls = critic.adjusted_tool_calls or plan.tool_calls

    if not final_tool_calls:
        return GovernorDecision(
            verdict="chat_only",
            final_steps=final_steps,
            final_tool_calls=[],
            final_response_mode=plan.response_type,
            notes="No tool calls; treating as explanatory response.",
        )

    return GovernorDecision(
        verdict="execute",
        final_steps=final_steps,
        final_tool_calls=final_tool_calls,
        final_response_mode=plan.response_type,
        notes="Plan approved for execution.",
    )


__all__ = ["decide"]
