"""Credential-safe Bailian OpenAI-compatible client with caching and retries."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import time
from typing import Any
from urllib import error, request

from .memory import append_jsonl


class BailianUnavailable(RuntimeError):
    pass


class BailianClient:
    def __init__(self, cache_path: str | Path, usage_path: str | Path,
                 tool_log: str | Path, model: str = "qwen3.7-plus",
                 temperature: float = 0.35, timeout: float = 90.0, retries: int = 3):
        self.model, self.temperature = model, float(temperature)
        self.timeout, self.retries = timeout, retries
        self.cache_path, self.usage_path, self.tool_log = Path(cache_path), Path(usage_path), Path(tool_log)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(self.cache_path, timeout=30)
        self._cache.execute("PRAGMA journal_mode=WAL")
        self._cache.execute("CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY,response TEXT,created_at TEXT)")
        self._cache.commit()

    @property
    def configured(self) -> bool:
        return bool(os.environ.get("DASHSCOPE_API_KEY") and os.environ.get("DASHSCOPE_BASE_URL"))

    def _endpoint(self) -> str:
        base = os.environ.get("DASHSCOPE_BASE_URL", "").rstrip("/")
        if not base:
            raise BailianUnavailable("DASHSCOPE_BASE_URL is not set")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def _price(self, input_tokens: int, output_tokens: int, endpoint: str) -> float:
        long_context = input_tokens > 256_000
        if "ap-southeast-1" in endpoint or "dashscope-intl" in endpoint:
            input_rate, output_rate = ((14.988, 44.965) if long_context else (3.747, 22.483))
        else:
            input_rate, output_rate = ((8.0, 48.0) if long_context else (2.0, 12.0))
        return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 response_schema: dict[str, Any] | None = None, enable_thinking: bool = False) -> dict[str, Any]:
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise BailianUnavailable("DASHSCOPE_API_KEY is not set")
        endpoint = self._endpoint()
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "enable_thinking": bool(enable_thinking),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "transagent_decision", "strict": True, "schema": response_schema}}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
        cache_key = hashlib.sha256(encoded).hexdigest()
        cached = self._cache.execute("SELECT response FROM responses WHERE key=?", (cache_key,)).fetchone()
        if cached:
            response = json.loads(cached[0])
            self._record_usage(response, endpoint, latency=0.0, retry_count=0, cache_hit=True, fallback=False)
            return response
        last_error = "request failed"
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                req = request.Request(endpoint, data=encoded, method="POST", headers={
                    "Authorization": "Bearer " + key, "Content-Type": "application/json"})
                with request.urlopen(req, timeout=self.timeout) as response_handle:
                    response = json.loads(response_handle.read().decode("utf-8"))
                latency = time.perf_counter() - started
                cached_response = self._strip_private_reasoning(response)
                self._cache.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?)", (
                    cache_key, json.dumps(cached_response, ensure_ascii=True), datetime.now(timezone.utc).isoformat()))
                self._cache.commit()
                self._record_usage(response, endpoint, latency, attempt, False, False)
                message = response.get("choices", [{}])[0].get("message", {})
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function", {})
                    append_jsonl(self.tool_log, {"time": datetime.now(timezone.utc).isoformat(),
                                                "tool": function.get("name"),
                                                "arguments": function.get("arguments", "{}"),
                                                "source": "qwen"})
                return response
            except (error.HTTPError, error.URLError, TimeoutError, socket.timeout,
                    json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
                if attempt < self.retries:
                    time.sleep(min(8.0, 0.75 * (2 ** attempt)))
        self._record_usage({}, endpoint, 0.0, self.retries, False, True)
        raise BailianUnavailable(f"Bailian request failed after finite retries: {last_error}")

    def _record_usage(self, response: dict[str, Any], endpoint: str, latency: float,
                      retry_count: int, cache_hit: bool, fallback: bool) -> None:
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        message = response.get("choices", [{}])[0].get("message", {})
        tool_names = [call.get("function", {}).get("name", "") for call in message.get("tool_calls") or []]
        record = {
            "time": datetime.now(timezone.utc).isoformat(), "model": response.get("model", self.model),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "latency_seconds": round(latency, 6), "retry_count": retry_count,
            "cache_hit": int(cache_hit),
            "tool_calls": ";".join(name for name in tool_names if name),
            "estimated_cost_cny": 0.0 if cache_hit else round(self._price(input_tokens, output_tokens, endpoint), 8),
            "fallback": int(fallback),
        }
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.usage_path.exists()
        with self.usage_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)

    def _strip_private_reasoning(self, value: Any) -> Any:
        private_keys = {"reasoning_content", "reasoning", "thinking", "chain_of_thought"}
        if isinstance(value, dict):
            return {key: self._strip_private_reasoning(item) for key, item in value.items()
                    if key not in private_keys}
        if isinstance(value, list):
            return [self._strip_private_reasoning(item) for item in value]
        return value

    def close(self) -> None:
        self._cache.close()
