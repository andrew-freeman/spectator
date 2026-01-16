from __future__ import annotations

import json
import math
import sqlite3
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class MemoryRecord:
    id: str
    ts: float
    text: str
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class SQLiteVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.init()

    def init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    meta TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    record_id TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES memory_records(id)
                )
                """
            )

    def add(self, records: Iterable[MemoryRecord], vectors: Iterable[list[float]]) -> None:
        record_list = list(records)
        vector_list = list(vectors)
        if len(record_list) != len(vector_list):
            raise ValueError("records and vectors length mismatch")
        with sqlite3.connect(self.path) as conn:
            for record, vector in zip(record_list, vector_list):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_records (id, ts, text, tags, meta)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.ts,
                        record.text,
                        json.dumps(record.tags),
                        json.dumps(record.meta),
                    ),
                )
                blob = _pack_vector(vector)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_vectors (record_id, dim, vector)
                    VALUES (?, ?, ?)
                    """,
                    (record.id, len(vector), blob),
                )

    def query(self, vector: list[float], top_k: int = 5) -> list[tuple[MemoryRecord, float]]:
        if top_k <= 0:
            return []
        query_norm = _vector_norm(vector)
        if query_norm == 0:
            return []
        results: list[tuple[MemoryRecord, float]] = []
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT r.id, r.ts, r.text, r.tags, r.meta, v.dim, v.vector
                FROM memory_records r
                JOIN memory_vectors v ON r.id = v.record_id
                """
            ).fetchall()
        for row in rows:
            record = MemoryRecord(
                id=row[0],
                ts=row[1],
                text=row[2],
                tags=json.loads(row[3]),
                meta=json.loads(row[4]),
            )
            dim = row[5]
            stored_vector = _unpack_vector(row[6], dim)
            if len(stored_vector) != len(vector):
                continue
            similarity = _cosine_similarity(vector, stored_vector, query_norm)
            results.append((record, similarity))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def _pack_vector(vector: list[float]) -> bytes:
    data = array("f", vector)
    return data.tobytes()


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    data = array("f")
    data.frombytes(blob)
    if dim:
        data = data[:dim]
    return list(data)


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(vector: list[float], stored: list[float], query_norm: float) -> float:
    denom = query_norm * _vector_norm(stored)
    if denom == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vector, stored))
    return dot / denom
