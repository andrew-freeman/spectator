"""Local model client supporting GGUF/transformer backends."""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import httpx

try:  # Optional dependency for llama.cpp based models
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class LocalLLMClient:
    """Wrapper around ``llama_cpp.Llama`` for local inference."""

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        n_ctx: int = 4096,
        server_url: Optional[str] = None,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.server_url = server_url
        self._client: Optional[httpx.Client] = None
        self._llm: Optional[Llama] = None

        if self.server_url:
            self._client = httpx.Client(timeout=60.0)
            return

        if Llama is None:
            raise ImportError("llama_cpp is required for LocalLLMClient")
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx)

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        if self.server_url:
            text = self._generate_remote(prompt, stop=stop)
        else:
            text = self._generate_local(prompt, stop=stop)

        LOGGER.debug(
            "LocalLLMClient prompt=%r response=%r",
            prompt[:200],
            text[:200],
        )
        return text

    def _generate_local(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        if not self._llm:
            raise RuntimeError("Local LLM not initialized")
        completion = self._llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=list(stop) if stop else None,
        )
        text = completion["choices"][0]["text"]
        return text.strip()

    def _generate_remote(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        if not self._client or not self.server_url:
            raise RuntimeError("Remote client not initialized")

        payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": list(stop) if stop else None,
        }

        retries = 5
        backoff = 1.0
        url = f"{self.server_url}/v1/chat/completions"
        for attempt in range(1, retries + 1):
            try:
                response = self._client.post(url, json=payload)
                if response.status_code in {429, 500, 502, 503}:
                    LOGGER.warning(
                        "Retrying request due to status %s (attempt %s/%s)",
                        response.status_code,
                        attempt,
                        retries,
                    )
                    if attempt == retries:
                        response.raise_for_status()
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return text.strip()
            except httpx.HTTPError:
                LOGGER.exception("Failed to fetch completion from remote server")
                if attempt == retries:
                    raise
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError("Failed to fetch completion from remote server")


__all__ = ["LocalLLMClient"]
