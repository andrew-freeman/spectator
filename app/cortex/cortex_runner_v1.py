# app/cortex/cortex_runner_v1.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.core.schemas import CortexOutput, Mode
from app.self_model import export_self_model_for_prompt, SelfModel
from app.world_model import export_world_model_for_prompt, WorldModel
from app.history.history_manager import HistoryManager  # optional for debugging

from .cortex_prompt_v1 import CORTEX_PROMPT_V1

LOGGER = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Protocol for the LLM client used by the Agent Cortex."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


@dataclass
class CortexInputBundle:
    """Lightweight container for what the Cortex sees."""

    last_cycle: Dict[str, Any]
    self_model: SelfModel
    world_model: WorldModel
    current_objectives: List[str]
    last_user_message: str = ""


class CortexRunner:
    """
    LLM-driven Agent Cortex V1.

    - Builds a strictly structured prompt from the last cycle + models.
    - Calls the LLM.
    - Parses the section-based output into a CortexOutput dataclass.

    This runner is SIDE-EFFECT FREE: it does not update any models itself.
    """

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        history: Optional[HistoryManager] = None,
    ) -> None:
        self._client = client
        self._history = history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, bundle: CortexInputBundle) -> CortexOutput:
        prompt = self._build_prompt(bundle)
        raw = self._client.generate(prompt, stop=None)
        if self._history is not None:
            self._history.append(
                {
                    "type": "cortex_raw",
                    "raw": raw,
                }
            )
        try:
            return self._parse_output(raw)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Cortex parsing failed, using fallback: %s", exc)
            return self._fallback_output(bundle)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self, bundle: CortexInputBundle) -> str:
        last_cycle_json = json.dumps(
            bundle.last_cycle or {},
            ensure_ascii=False,
            indent=2,
        )
        self_model_json = json.dumps(
            export_self_model_for_prompt(bundle.self_model),
            ensure_ascii=False,
            indent=2,
        )
        world_model_json = json.dumps(
            export_world_model_for_prompt(bundle.world_model),
            ensure_ascii=False,
            indent=2,
        )
        objectives_json = json.dumps(
            bundle.current_objectives or [],
            ensure_ascii=False,
            indent=2,
        )
        last_user_message = bundle.last_user_message or ""

        prompt = CORTEX_PROMPT_V1
        prompt = prompt.replace("<<LAST_CYCLE_JSON>>", last_cycle_json)
        prompt = prompt.replace("<<SELF_MODEL_JSON>>", self_model_json)
        prompt = prompt.replace("<<WORLD_MODEL_JSON>>", world_model_json)
        prompt = prompt.replace("<<OBJECTIVES_LIST_JSON>>", objectives_json)
        prompt = prompt.replace("<<LAST_USER_MESSAGE>>", last_user_message)
        return prompt

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_output(self, raw: str) -> CortexOutput:
        """
        Parse the section-based Cortex response into a CortexOutput.

        Expected sections:
        - #SUMMARY:
        - #NEXT_OBJECTIVES:
        - #MODE_HINT:
        - #FORCE_ACTION:
        - #SAFETY_BIAS:
        - #SELF_NOTES:
        - #WORLD_NOTES:
        """

        # Normalise line endings and strip leading/trailing whitespace
        text = (raw or "").replace("\r\n", "\n").strip()

        sections = self._split_sections(text)

        next_objectives = self._parse_bullet_list(sections.get("NEXT_OBJECTIVES", ""))
        self_notes = self._parse_bullet_list(sections.get("SELF_NOTES", ""))
        world_notes = self._parse_bullet_list(sections.get("WORLD_NOTES", ""))

        mode_hint_str = sections.get("MODE_HINT", "").strip().lower()
        mode_hint: Optional[Mode]
        if mode_hint_str in {"chat", "knowledge", "world_query", "world_control"}:
            mode_hint = mode_hint_str  # type: ignore[assignment]
        else:
            mode_hint = None  # "unchanged" or invalid

        force_action_str = sections.get("FORCE_ACTION", "").strip().lower()
        force_action = force_action_str == "true"

        safety_bias_str = sections.get("SAFETY_BIAS", "").strip().lower()
        if safety_bias_str not in {"conservative", "normal", "aggressive"}:
            safety_bias_str = "normal"

        if not next_objectives:
            next_objectives = []

        return CortexOutput(
            next_objectives=next_objectives,
            mode_hint=mode_hint,
            force_action=force_action,
            safety_bias=safety_bias_str,  # type: ignore[arg-type]
            self_notes=self_notes,
            world_notes=world_notes,
        )

    def _split_sections(self, text: str) -> Dict[str, str]:
        """
        Turn a section-based text into a dict like:
        { "SUMMARY": "...", "NEXT_OBJECTIVES": "...", ... }.
        """

        sections: Dict[str, str] = {}
        current_key: Optional[str] = None
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer, current_key
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            buffer = []

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") and stripped.endswith(":"):
                # New section header
                flush()
                header_name = stripped.strip("#:").strip().upper()
                current_key = header_name
                continue
            buffer.append(line)

        flush()
        return sections

    def _parse_bullet_list(self, block: str) -> List[str]:
        """
        Parse lines starting with '-' into a clean list of items.
        """
        items: List[str] = []
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("-"):
                item = stripped.lstrip("-").strip()
                if item:
                    items.append(item)
        return items

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_output(self, bundle: CortexInputBundle) -> CortexOutput:
        """
        Conservative default when parsing fails.
        """
        return CortexOutput(
            next_objectives=list(bundle.current_objectives or []),
            mode_hint=None,
            force_action=False,
            safety_bias="conservative",
            self_notes=["Cortex fallback: kept existing objectives."],
            world_notes=[],
        )


__all__ = ["CortexRunner", "CortexInputBundle"]