"""Deterministic LLMs.

ScriptedLLM: replays a fixed list of responses. Used by the tests to drive exact paths
(tool call -> reply, bad reply -> rewrite, outage -> fallback).

DemoLLM: a tiny rule-based stand-in so `docker compose up` works without an API key.
It exists to show the plumbing (tools, memory, guards, handoff), not to be smart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ..text import detect_language, normalize
from .base import LLMError, LLMResponse, ToolCall

Step = LLMResponse | Exception | Callable[[list[dict[str, Any]]], LLMResponse]


class ScriptedLLM:
    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.steps:
            raise LLMError("ScriptedLLM ran out of steps")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step


def reply(text: str, tokens: tuple[int, int] = (100, 20)) -> LLMResponse:
    return LLMResponse(
        content=text, model="scripted", input_tokens=tokens[0], output_tokens=tokens[1]
    )


def tool(name: str, call_id: str = "call_1", **arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        model="scripted",
        input_tokens=100,
        output_tokens=10,
    )


_HANDOFF_WORDS = ["נציג", "בן אדם", "human", "agent", "موظف", "انسان", "شخص"]
_BOOK_WORDS = ["להזמין", "הזמנה", "לשריין", "book", "reserve", "احجز", "حجز"]

_TEXT = {
    "hebrew": {
        "item": "{name}: {price} ₪ {unit}.",
        "ask": " לאיזה תאריך תרצו?",
        "none": "לא מצאתי את זה אצלנו. אפשר לפרט מה מחפשים?",
        "lead": "מעולה, רשמתי את הבקשה ונציג יחזור אליכם לאישור.",
        "handoff": "מעביר אתכם לנציג, הוא יענה כאן בהקדם.",
    },
    "arabic": {
        "item": "{name}: {price} شيكل {unit}.",
        "ask": " لأي تاريخ؟",
        "none": "لم أجد ذلك لدينا. ممكن توضح ما الذي تبحث عنه؟",
        "lead": "ممتاز، سجلت طلبك وسيتواصل معك موظف للتأكيد.",
        "handoff": "سأحولك إلى موظف، سيرد عليك هنا قريبا.",
    },
    "english": {
        "item": "{name}: {price} ILS {unit}.",
        "ask": " Which date works for you?",
        "none": "I couldn't find that. Could you tell me a bit more?",
        "lead": "Great, I've logged your request and a team member will confirm shortly.",
        "handoff": "I'm passing you to a team member, they'll reply here shortly.",
    },
}


class DemoLLM:
    model = "demo-rules"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        last = messages[-1]
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lang = detect_language(user_text, default="hebrew")
        t = _TEXT[lang]

        if last["role"] == "tool":
            result = json.loads(last["content"])
            name = result.get("tool")
            if name == "handoff_to_human":
                return self._reply(t["handoff"])
            if name == "create_lead":
                return self._reply(t["lead"])
            items = result.get("items") or []
            if not items:
                return self._reply(t["none"])
            lines = [t["item"].format(**i) for i in items[:2]]
            return self._reply(" ".join(lines) + t["ask"])

        text = normalize(user_text)
        if any(normalize(w) in text for w in _HANDOFF_WORDS):
            return self._tool("handoff_to_human", reason="customer asked for a person")
        if any(normalize(w) in text for w in _BOOK_WORDS) or re.search(
            r"\d{1,2}[./]\d{1,2}", user_text
        ):
            return self._tool("create_lead", interest=user_text[:120])
        return self._tool("search_catalog", query=user_text)

    def _reply(self, text: str) -> LLMResponse:
        return LLMResponse(content=text, model=self.model, input_tokens=0, output_tokens=0)

    def _tool(self, name: str, **args: Any) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id=f"demo_{name}", name=name, arguments=args)],
            model=self.model,
        )
