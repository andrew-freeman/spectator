from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbedder:
    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
            for idx in range(0, len(digest), 4):
                chunk = digest[idx : idx + 4]
                if len(chunk) < 4:
                    continue
                number = int.from_bytes(chunk, "big", signed=False)
                values.append(number / 2**32)
                if len(values) >= self.dim:
                    break
            counter += 1
        return _normalize(values)


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]
