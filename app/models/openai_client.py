"""HTTP client for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, Iterable, Optional

import httpx


LOGGER = logging.getLogger(__name__)


class OpenAIClientError(RuntimeError):
    """Custom error raised when the OpenAI-compatible endpoint fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_payload: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_payload = response_payload

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"message": str(self)}
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.response_payload is not None:
            payload["response"] = self.response_payload
        return payload


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

        max_retries = 5
        base_delay = 0.5
        jitter = 0.25
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                LOGGER.debug(
                    "Dispatching chat completion request (attempt %d/%d)", attempt + 1, max_retries
                )
                response = self._client.post("/chat/completions", json=payload)
                if response.status_code == 429:
                    wait = base_delay * (2**attempt) + random.uniform(0, jitter)
                    LOGGER.warning(
                        "Rate limit hit, retry %d/%d, waiting %.2fs", attempt + 1, max_retries, wait
                    )
                    time.sleep(wait)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:  # pragma: no cover - defensive network guard
                    last_error = OpenAIClientError(
                        "OpenAI-compatible endpoint returned an error",
                        status_code=exc.response.status_code if exc.response else None,
                        response_payload=_safe_response_payload(exc.response),
                    )
                    LOGGER.error("LLM request failed with HTTP %s", getattr(exc.response, "status_code", "?"))
                    continue

                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError) as exc:  # pragma: no cover - network response guard
                    last_error = RuntimeError(f"Unexpected response payload: {data}")
                    LOGGER.error("LLM response payload missing expected fields: %s", exc)
            except httpx.HTTPError as exc:
                last_error = exc
                LOGGER.error(
                    "LLM request failed due to HTTP error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensive network guard
                last_error = exc
                LOGGER.error(
                    "LLM request failed due to unexpected error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )

        LOGGER.error("LLM request failed after %d retries", max_retries)
        error_payload: Dict[str, Any]
        if isinstance(last_error, OpenAIClientError):
            error_payload = {"error": last_error.to_payload(), "attempts": max_retries}
        else:
            error_payload = {
                "error": {
                    "message": "LLM request failed after retries",
                    "details": str(last_error) if last_error else "Unknown error",
                },
                "attempts": max_retries,
            }
        return json.dumps(error_payload)


def _safe_response_payload(response: Optional[httpx.Response]) -> Optional[Any]:
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:  # pragma: no cover - defensive network guard
        return response.text


__all__ = ["OpenAIClient", "OpenAIClientError"]
