from __future__ import annotations

from dataclasses import dataclass

from spectator.memory.embeddings import Embedder
from spectator.memory.vector_store import SQLiteVectorStore


@dataclass(slots=True)
class MemoryContext:
    store: SQLiteVectorStore
    embedder: Embedder

