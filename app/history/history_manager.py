import json
import os
import time
from typing import List, Dict, Any


class HistoryManager:
    def __init__(self, base_dir: str = "session_logs"):
        os.makedirs(base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(base_dir, f"session_{timestamp}.json")
        self._log: List[Dict[str, Any]] = []

    def append(self, entry: Dict[str, Any]):
        self._log.append(entry)
        with open(self.path, "w") as f:
            json.dump(self._log, f, indent=2)

    def load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except:
            return self._log
