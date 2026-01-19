from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def system_time_handler() -> Any:
    def handler(_args: dict[str, Any]) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now().astimezone()
        return {
            "utc": now_utc.isoformat().replace("+00:00", "Z"),
            "local": now_local.isoformat(),
            "epoch_s": now_utc.timestamp(),
        }

    return handler
