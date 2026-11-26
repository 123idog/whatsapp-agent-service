import json

import httpx
import pytest

from agent_service.llm import LLMError, OpenAICompatLLM


def ok_body(model="m", content="שלום", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [{"message": msg}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 7},
    }


def client(handler, **kw):
    sleeps = []
    llm = OpenAICompatLLM(
        "https://llm.test/v1",
        "k",
        "primary",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kw,
    )
    return llm, sleeps


def test_parses_tool_calls_and_usage():
    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "primary" and body["tools"]
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json=ok_body(
                content=None,
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_catalog", "arguments": '{"query": "סיור"}'},
                    }
                ],
            ),
        )

    llm, _ = client(handler)
    r = llm.chat(
        [{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "t"}}]
    )
    assert r.tool_calls[0].name == "search_catalog"
    assert r.tool_calls[0].arguments == {"query": "סיור"}
    assert (r.input_tokens, r.output_tokens) == (50, 7)


def test_retries_429_honouring_retry_after():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return httpx.Response(200, json=ok_body())

    llm, sleeps = client(handler)
    assert llm.chat([], []).content == "שלום"
    assert len(calls) == 3 and sleeps == [2.0, 2.0]


def test_falls_back_to_second_model_after_retries():
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        if model == "primary":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=ok_body(model="backup"))

    llm, _ = client(handler, fallback_model="backup", max_retries=2)
    r = llm.chat([], [])
    assert r.model == "backup"
    assert seen == ["primary"] * 3 + ["backup"]


def test_no_retry_on_400():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    llm, _ = client(handler)
    with pytest.raises(LLMError):
        llm.chat([], [])
    assert len(calls) == 1


def test_timeouts_are_retried_then_raise():
    def handler(req):
        raise httpx.ReadTimeout("timed out", request=req)

    llm, sleeps = client(handler, max_retries=2)
    with pytest.raises(LLMError, match="all models failed"):
        llm.chat([], [])
    assert len(sleeps) == 2


def test_malformed_tool_arguments_do_not_crash():
    def handler(req):
        return httpx.Response(
            200,
            json=ok_body(
                content=None,
                tool_calls=[
                    {"id": "c1", "function": {"name": "search_catalog", "arguments": "{not json"}}
                ],
            ),
        )

    llm, _ = client(handler)
    assert llm.chat([], []).tool_calls[0].arguments == {"_unparsed": "{not json"}
