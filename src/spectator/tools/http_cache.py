from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CachedHttpResponse:
    url: str
    status: int
    text: str
    stored_ts: float


class HttpCache:
    def __init__(self, path: Path, ttl_s: float) -> None:
        self._path = path
        self._ttl_s = ttl_s
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS http_cache (
                    url TEXT PRIMARY KEY,
                    status INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    stored_ts REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, url: str) -> CachedHttpResponse | None:
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                "SELECT url, status, text, stored_ts FROM http_cache WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None:
            return None
        stored_ts = float(row[3])
        if time.time() - stored_ts > self._ttl_s:
            return None
        return CachedHttpResponse(url=row[0], status=int(row[1]), text=row[2], stored_ts=stored_ts)

    def set(self, url: str, status: int, text: str) -> None:
        stored_ts = time.time()
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT INTO http_cache (url, status, text, stored_ts)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    status = excluded.status,
                    text = excluded.text,
                    stored_ts = excluded.stored_ts
                """,
                (url, status, text, stored_ts),
            )
            conn.commit()
