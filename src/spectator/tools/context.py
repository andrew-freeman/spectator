from __future__ import annotations

from dataclasses import dataclass

from spectator.core.types import State
from spectator.tools.settings import ToolSettings


@dataclass(slots=True)
class ToolContext:
    state: State
    settings: ToolSettings
