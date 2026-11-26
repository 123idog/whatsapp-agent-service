"""Client for any OpenAI-compatible chat-completions endpoint.

One adapter covers OpenAI, Gemini (OpenAI-compatible endpoint), OpenRouter, Ollama, vLLM.
Production behaviour, not SDK sugar:
- retries with exponential backoff + jitter on 429 / 5xx / timeouts, honouring Retry-After
- a fallback model when the primary is still failing after its retries
- no retry on 4xx other than 429: a bad request will not get better by repeating it
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from .base import LLMError, LLMResponse, ToolCall

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        fallback_model: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.temperature = temperature
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last_error: Exception | None = None
        for model in models:
            try:
                return self._chat_with_retries(model, messages, tools)
            except LLMError as e:
                last_error = e
                log.warning(
                    "model failed, trying fallback", extra={"model": model, "error": str(e)}
                )
        raise LLMError(f"all models failed: {last_error}")

    def _chat_with_retries(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as e:  # timeouts, connection resets
                error: str = f"{type(e).__name__}: {e}"
                retry_after = None
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), model)
                error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(error)
                retry_after = _retry_after_seconds(resp)
            if attempt == self.max_retries:
                raise LLMError(error)
            delay = retry_after if retry_after is not None else (2**attempt) + random.random()
            log.info("llm retry", extra={"model": model, "attempt": attempt + 1, "error": error})
            self._sleep(min(delay, 30.0))
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(data: dict[str, Any], model: str) -> LLMResponse:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        calls = []
        for tc in message.get("tool_calls") or []:
            raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            calls.append(ToolCall(id=tc.get("id", ""), name=tc["function"]["name"], arguments=args))
        usage = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=calls,
            model=data.get("model") or model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
