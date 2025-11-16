from __future__ import annotations
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class StateStore:
    """Simple persistent JSONL store for system state & cycle history."""

    def __init__(self, base_path: str = "storage"):
        self.base_path = base_path
        self.state_file = os.path.join(base_path, "latest_state.json")
        self.history_file = os.path.join(base_path, "history.jsonl")
        self.lock = threading.Lock()

        os.makedirs(base_path, exist_ok=True)

    def save_latest(self, state: Dict[str, Any]):
        """Overwrite latest_state.json"""
        with self.lock:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def append_history(self, record: Dict[str, Any]):
        """Append historical record to JSONL file"""
        record["timestamp"] = datetime.utcnow().isoformat()
        with self.lock:
            with open(self.history_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    def load_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load last N entries"""
        if not os.path.exists(self.history_file):
            return []

        with open(self.history_file, "r") as f:
            lines = f.readlines()

        return [json.loads(line) for line in lines[-limit:]]


GLOBAL_STATE_STORE = StateStore()

__all__ = ["GLOBAL_STATE_STORE", "StateStore"]
