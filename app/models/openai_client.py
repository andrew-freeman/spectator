"""HTTP client for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import httpx


class OpenAIClient:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}"})

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a JSON-only reasoning engine."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = list(stop)
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - network response guard
            raise RuntimeError(f"Unexpected response payload: {data}") from exc


__all__ = ["OpenAIClient"]
