"""Local long-term memory components."""

from spectator.memory.embeddings import Embedder, HashEmbedder
from spectator.memory.retrieval import format_retrieval_block, retrieve
from spectator.memory.vector_store import MemoryRecord, SQLiteVectorStore

__all__ = [
    "Embedder",
    "HashEmbedder",
    "MemoryRecord",
    "SQLiteVectorStore",
    "format_retrieval_block",
    "retrieve",
]
