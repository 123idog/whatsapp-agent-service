"""The tool-calling loop: model -> tools -> model, bounded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLM
from .tools import TOOL_SCHEMAS, ToolContext, execute


@dataclass
class AgentRun:
    text: str | None
    models: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0


def run_agent(
    llm: LLM,
    messages: list[dict[str, Any]],
    ctx: ToolContext,
    *,
    max_steps: int = 5,
) -> AgentRun:
    """Run until the model answers in text or max_steps is hit.

    `messages` is extended in place with the tool exchange, so a rewrite attempt after a
    guard failure sees what the tools returned the first time.
    Raises LLMError if the model is unreachable.
    """
    run = AgentRun(text=None)
    for _ in range(max_steps):
        resp = llm.chat(messages, TOOL_SCHEMAS)
        run.steps += 1
        run.input_tokens += resp.input_tokens
        run.output_tokens += resp.output_tokens
        if resp.model and resp.model not in run.models:
            run.models.append(resp.model)

        if not resp.tool_calls:
            run.text = (resp.content or "").strip()
            return run

        messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": _json(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ],
            }
        )
        for tc in resp.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": execute(ctx, tc.name, tc.arguments),
                }
            )
    return run  # text=None: the model kept calling tools; the pipeline falls back


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
