from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from spectator.backends.registry import register_backend
from spectator.core.tracing import TraceEvent
from spectator.prompts import load_prompt

_DEFAULT_LLAMA_RULES_PROMPT = "system/llama_rules.txt"
_ENV_LLAMA_RULES_PROMPT = "SPECTATOR_LLAMA_RULES_PROMPT"
_ENV_LLAMA_LOG_PAYLOAD = "SPECTATOR_LLAMA_LOG_PAYLOAD"
_ENV_LLAMA_LOG_DIR = "SPECTATOR_LLAMA_LOG_DIR"


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class LlamaServerBackend:
    base_url: str = os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080")
    timeout_s: float = _env_float("LLAMA_SERVER_TIMEOUT_S", 60.0)
    api_key: str | None = os.getenv("LLAMA_SERVER_API_KEY")
    model: str | None = os.getenv("LLAMA_SERVER_MODEL")
    supports_messages: bool = True
    reset_slot: bool = _env_bool("LLAMA_SERVER_RESET_SLOT", False)
    slot_id: int = _env_int("LLAMA_SERVER_SLOT_ID", 0)
    _reset_run_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    '''
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
    '''

    def _ensure_system_rules(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
    ) -> list[dict[str, Any]]:
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    messages[i] = {"role": "system", "content": f"{system_text}\n\n{content}"}
                else:
                    messages[i] = {"role": "system", "content": system_text}
                return messages
        return [{"role": "system", "content": system_text}, *messages]

    def _ensure_system_rules(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
    ) -> list[dict[str, Any]]:
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    messages[i] = {"role": "system", "content": f"{system_text}\n\n{content}"}
                else:
                    messages[i] = {"role": "system", "content": system_text}
                return messages
        return [{"role": "system", "content": system_text}, *messages]


    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        options = dict(params)
        options.pop("role", None)
        options.pop("stream_callback", None)

        # NEW: accept upstream system_prompt
        upstream_system = options.pop("system_prompt", None)
        if upstream_system is not None and not isinstance(upstream_system, str):
            upstream_system = str(upstream_system)

        messages = options.pop("messages", None)
        model = options.pop("model", self.model)

        options.setdefault("temperature", 0)
        options.setdefault("top_p", 1)
        options.setdefault("max_tokens", 512)
        options.setdefault("seed", 7)

        # Base rules always exist
        base_rules = _build_system_rules(model)
        merged_system = base_rules if not upstream_system else f"{base_rules}\n\n{upstream_system}"

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Ensure system is present / prefixed
        messages = self._ensure_system_rules(messages, merged_system)

        payload = {"messages": messages}
        if model:
            payload["model"] = model
        payload.setdefault("cache_prompt", False)
        payload.update(options)
        return payload

    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        options = dict(params)
        options.pop("role", None)
        options.pop("stream_callback", None)

        # NEW: accept upstream system_prompt
        upstream_system = options.pop("system_prompt", None)
        if upstream_system is not None and not isinstance(upstream_system, str):
            upstream_system = str(upstream_system)

        messages = options.pop("messages", None)
        model = options.pop("model", self.model)

        options.setdefault("temperature", 0)
        options.setdefault("top_p", 1)
        options.setdefault("max_tokens", 512)
        options.setdefault("seed", 7)

        # Base rules always exist
        base_rules = _build_system_rules(model)
        merged_system = base_rules if not upstream_system else f"{base_rules}\n\n{upstream_system}"

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Ensure system is present / prefixed
        messages = self._ensure_system_rules(messages, merged_system)

        payload = {"messages": messages}
        if model:
            payload["model"] = model
        payload.setdefault("cache_prompt", False)
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

    @staticmethod
    def _log_payload(payload: dict[str, Any]) -> None:
        if not _env_bool(_ENV_LLAMA_LOG_PAYLOAD, False):
            return
        pretty_payload = json.dumps(payload, indent=2)
        log_dir = os.getenv(_ENV_LLAMA_LOG_DIR)
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            for _ in range(5):
                timestamp = time.time_ns()
                filename = f"llama_payload_{timestamp}_{pid}.json"
                payload_path = log_path / filename
                try:
                    with payload_path.open("x", encoding="utf-8") as handle:
                        handle.write(pretty_payload)
                    return
                except FileExistsError:
                    continue
            logging.getLogger(__name__).warning(
                "Failed to write llama payload log after multiple attempts."
            )
            return
        logging.getLogger(__name__).info("Llama request payload:\n%s", pretty_payload)

    def _open_stream(self, url: str, payload: dict[str, Any]) -> Iterator[str]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        response = urllib.request.urlopen(request, timeout=self.timeout_s)
        try:
            for raw_line in response:
                yield raw_line.decode("utf-8")
        finally:
            response.close()

    def reset_slot_cache(self, run_id: str | None = None, tracer=None) -> None:
        if not self.reset_slot:
            return
        token = run_id or "default"
        if token in self._reset_run_ids:
            return
        self._reset_run_ids.add(token)
        url = f"{self.base_url.rstrip('/')}/slots/{self.slot_id}?action=erase"
        try:
            request = urllib.request.Request(url, data=b"", headers=self._headers(), method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout_s):
                return
        except Exception as exc:  # noqa: BLE001
            if tracer is not None:
                tracer.write(
                    TraceEvent(
                        ts=time.time(),
                        kind="warning",
                        data={
                            "backend": "llama",
                            "message": "Failed to reset llama slot cache.",
                            "error": str(exc),
                        },
                    )
                )

    def complete(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        payload = self._build_payload(prompt, params)
        self._log_payload(payload)
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
