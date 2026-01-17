from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from spectator.backends.registry import register_backend
from spectator.prompts import load_prompt

_DEFAULT_LLAMA_RULES_PROMPT = "system/llama_rules.txt"
_ENV_LLAMA_RULES_PROMPT = "SPECTATOR_LLAMA_RULES_PROMPT"


def _load_llama_rules() -> str:
    rel_path = os.getenv(_ENV_LLAMA_RULES_PROMPT, _DEFAULT_LLAMA_RULES_PROMPT)
    return load_prompt(rel_path)


def _build_system_rules(model: str | None) -> str:
    model_line = (
        f"The underlying model is {model}."
        if model
        else "The underlying model is unknown."
    )
    return f"{_load_llama_rules()} {model_line}"


def build_system_rules(model: str | None) -> str:
    return _build_system_rules(model)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class LlamaServerBackend:
    base_url: str = os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080")
    timeout_s: float = _env_float("LLAMA_SERVER_TIMEOUT_S", 60.0)
    api_key: str | None = os.getenv("LLAMA_SERVER_API_KEY")
    model: str | None = os.getenv("LLAMA_SERVER_MODEL")
    supports_messages: bool = True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        options = dict(params)
        options.pop("role", None)
        options.pop("stream_callback", None)
        messages = options.pop("messages", None)
        model = options.pop("model", self.model)
        options.setdefault("temperature", 0)
        options.setdefault("top_p", 1)
        options.setdefault("max_tokens", 512)
        options.setdefault("seed", 7)
        if messages is None:
            messages = [
                {"role": "system", "content": _build_system_rules(model)},
                {"role": "user", "content": prompt},
            ]
        payload = {"messages": messages}
        if model:
            payload["model"] = model
        if "cache_prompt" not in options:
            payload["cache_prompt"] = False
        payload.update(options)
        return payload

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
        return ""

    @staticmethod
    def _iter_sse_lines(lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("data:"):
                yield stripped[len("data:") :].strip()

    def _open_stream(self, url: str, payload: dict[str, Any]) -> Iterator[str]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        response = urllib.request.urlopen(request, timeout=self.timeout_s)
        try:
            for raw_line in response:
                yield raw_line.decode("utf-8")
        finally:
            response.close()

    def complete(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        payload = self._build_payload(prompt, params)
        stream = bool(payload.get("stream"))
        stream_callback = params.get("stream_callback")
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        if not stream:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
            return self._extract_content(json.loads(body))

        raw_parts: list[str] = []
        for data in self._iter_sse_lines(self._open_stream(url, payload)):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = ""
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("delta")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            delta = content
                    elif isinstance(first.get("text"), str):
                        delta = first["text"]
            if delta:
                raw_parts.append(delta)
                if callable(stream_callback):
                    stream_callback(delta)
        return "".join(raw_parts)


register_backend("llama", LlamaServerBackend)
