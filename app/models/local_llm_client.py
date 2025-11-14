"""Local model client supporting GGUF/transformer backends."""

from __future__ import annotations

from typing import Iterable, Optional

try:  # Optional dependency for llama.cpp based models
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


class LocalLLMClient:
    """Wrapper around ``llama_cpp.Llama`` for local inference."""

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        n_ctx: int = 4096,
    ) -> None:
        if Llama is None:
            raise ImportError("llama_cpp is required for LocalLLMClient")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx)

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        completion = self._llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=list(stop) if stop else None,
        )
        text = completion["choices"][0]["text"]
        return text.strip()


__all__ = ["LocalLLMClient"]
