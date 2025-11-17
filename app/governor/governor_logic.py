"""
V3 Governor: Mediates between planner, critic, and executor.
Applies strict validation rules and deterministic safety checks.
"""
from __future__ import annotations

from typing import Dict, List

from app.core.schemas import CriticOutput, GovernorDecision, PlannerPlan, ToolCall
from app.core.tool_registry import READ_TOOLS, CONTROL_TOOLS


def arbitrate(
    plan: PlannerPlan,
    critic: CriticOutput,
    *,
    mode: str,
    context: Dict[str, object] | None = None,
) -> GovernorDecision:
    """
    Decide whether the proposed plan and its tool calls are safe to execute.

    The Governor enforces:
    - Critic risk gates (unsafe/high → reject)
    - Mode rules (chat/knowledge/query/control)
    - Tool registry enforcement (read-only vs control tools)
    - No silent fallbacks
    """

    context = context or {}

    # Determine final mode (planner-selected mode is authoritative unless overridden)
    safe_mode = mode if mode in {"chat", "knowledge", "world_query", "world_control"} else plan.mode

    # Metadata included in the GovernorDecision for debugging and auditing:
    metadata = {
        "mode": safe_mode,
        "critic_risk": critic.risk_level,
        "critic_confidence": critic.confidence,
        "issues": critic.detected_issues,
    }
    metadata.update({k: v for k, v in context.items() if isinstance(k, str)})

    # =====================================================
    # 1. Critic Risk Gating
    # =====================================================
    if critic.risk_level in {"unsafe"}:
        return GovernorDecision(
            verdict="reject",
            rationale=critic.notes or "Critic classified the plan as UNSAFE.",
            final_tool_calls=[],
            metadata=metadata,
        )

    if critic.risk_level in {"high"}:
        return GovernorDecision(
            verdict="reject",
            rationale=critic.notes or "Critic classified the plan as HIGH risk.",
            final_tool_calls=[],
            metadata=metadata,
        )

    # =====================================================
    # 2. Mode: CHAT
    # =====================================================
    if safe_mode == "chat":
        return GovernorDecision(
            verdict="approve",
            rationale="Chat mode: responder-only, no tools permitted.",
            final_tool_calls=[],
            metadata=metadata,
        )

    # =====================================================
    # 3. Mode: KNOWLEDGE
    # =====================================================
    if safe_mode == "knowledge":
        return GovernorDecision(
            verdict="approve",
            rationale="Knowledge mode: pure reasoning, no tools allowed.",
            final_tool_calls=[],
            metadata=metadata,
        )

    # =====================================================
    # 4. Mode: WORLD_QUERY (must use read-only tools)
    # =====================================================
    if safe_mode == "world_query" or context.get("query_mode"):
        valid_calls: List[ToolCall] = []

        for call in plan.tool_calls:
            if call.name not in READ_TOOLS:
                return GovernorDecision(
                    verdict="reject",
                    rationale=f"Query mode violation: tool {call.name!r} is not a READ tool.",
                    final_tool_calls=[],
                    metadata=metadata,
                )
            valid_calls.append(call)

        if not valid_calls:
            return GovernorDecision(
                verdict="request_more_data",
                rationale="Query mode: no read tools provided.",
                final_tool_calls=[],
                metadata=metadata,
            )

        return GovernorDecision(
            verdict="query_mode",
            rationale="Approved read-only query.",
            final_tool_calls=valid_calls,
            metadata=metadata,
        )

    # =====================================================
    # 5. Mode: WORLD_CONTROL (must use at least 1 control tool)
    # =====================================================
    if safe_mode == "world_control":
        if not plan.tool_calls:
            return GovernorDecision(
                verdict="request_more_data",
                rationale="Control mode: plan contains no tool calls.",
                final_tool_calls=[],
                metadata=metadata,
            )

        valid_calls: List[ToolCall] = []

        for call in plan.tool_calls:
            if call.name not in CONTROL_TOOLS:
                return GovernorDecision(
                    verdict="reject",
                    rationale=f"Control mode violation: tool {call.name!r} is not a CONTROL tool.",
                    final_tool_calls=[],
                    metadata=metadata,
                )
            valid_calls.append(call)

        return GovernorDecision(
            verdict="approve",
            rationale="Control plan approved.",
            final_tool_calls=valid_calls,
            metadata=metadata,
        )

    # =====================================================
    # 6. Fallback (should not normally trigger)
    # =====================================================
    return GovernorDecision(
        verdict="approve",
        rationale="Default approval (fallback).",
        final_tool_calls=[],
        metadata=metadata,
    )