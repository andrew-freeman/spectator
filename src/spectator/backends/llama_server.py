from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

from spectator.backends.registry import register_backend
from spectator.core.tracing import TraceEvent
from spectator.prompts import load_prompt
from spectator.utils import test_artifacts

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
        stream = bool(payload.get("stream"))
        stream_callback = params.get("stream_callback")
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers = self._headers()
        context = test_artifacts.get_test_context()
        pid = os.getpid()
        seq = test_artifacts.next_sequence() if context.enabled else None
        request_info = {
            "url": url,
            "method": "POST",
            "headers": headers,
            "payload": payload,
        }
        if context.enabled and seq is not None and context.outdir and context.case_id:
            requests_dir, responses_dir, meta_dir = test_artifacts.artifact_paths(
                context.outdir, context.case_id
            )
            test_artifacts.ensure_dirs(requests_dir, responses_dir, meta_dir)
            redacted_request = test_artifacts.maybe_redact(request_info, context.redact)
            request_bytes = test_artifacts.json_bytes(redacted_request)
            request_path = requests_dir / f"{seq:04d}__pid{pid}.json"
            test_artifacts.write_bytes(request_path, request_bytes)
            request_hash = test_artifacts.sha256_bytes(request_bytes)
        else:
            requests_dir = responses_dir = meta_dir = None
            request_hash = None

        if not stream:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
                status_code = response.status
            latency_ms = int((time.monotonic() - start) * 1000)
            if (
                context.enabled
                and seq is not None
                and responses_dir is not None
                and meta_dir is not None
                and context.outdir
                and context.case_id
            ):
                response_bytes = body.encode("utf-8")
                response_path = responses_dir / f"{seq:04d}__pid{pid}.json"
                test_artifacts.write_bytes(response_path, response_bytes)
                response_hash = test_artifacts.sha256_bytes(response_bytes)
                meta = {
                    "timestamp": time.time(),
                    "url": url,
                    "method": "POST",
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "backend": "llama",
                    "model": payload.get("model"),
                    "session_id": context.session_id,
                    "case_id": context.case_id,
                    "request_sha256": request_hash,
                    "response_sha256": response_hash,
                }
                meta_path = meta_dir / f"{seq:04d}__pid{pid}.meta.json"
                test_artifacts.write_json(meta_path, meta)
            return self._extract_content(json.loads(body))

        raw_parts: list[str] = []
        response_lines: list[str] = []
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            status_code = response.status
            done = False
            for raw_line in response:
                decoded_line = raw_line.decode("utf-8")
                response_lines.append(decoded_line)
                for sse_payload in self._iter_sse_lines([decoded_line]):
                    if sse_payload == "[DONE]":
                        done = True
                        break
                    try:
                        sse_data = json.loads(sse_payload)
                    except json.JSONDecodeError:
                        continue
                    delta = ""
                    choices = sse_data.get("choices")
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
                if done:
                    break
        latency_ms = int((time.monotonic() - start) * 1000)
        if (
            context.enabled
            and seq is not None
            and responses_dir is not None
            and meta_dir is not None
            and context.outdir
            and context.case_id
        ):
            response_text = "".join(response_lines)
            response_bytes = response_text.encode("utf-8")
            response_path = responses_dir / f"{seq:04d}__pid{pid}.json"
            test_artifacts.write_bytes(response_path, response_bytes)
            response_hash = test_artifacts.sha256_bytes(response_bytes)
            meta = {
                "timestamp": time.time(),
                "url": url,
                "method": "POST",
                "status_code": status_code,
                "latency_ms": latency_ms,
                "backend": "llama",
                "model": payload.get("model"),
                "session_id": context.session_id,
                "case_id": context.case_id,
                "request_sha256": request_hash,
                "response_sha256": response_hash,
            }
            meta_path = meta_dir / f"{seq:04d}__pid{pid}.meta.json"
            test_artifacts.write_json(meta_path, meta)
        return "".join(raw_parts)


register_backend("llama", LlamaServerBackend)
