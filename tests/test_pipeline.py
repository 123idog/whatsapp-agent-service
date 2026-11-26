import threading
import time

import pytest

from agent_service.llm import LLMError
from agent_service.llm.scripted import reply, tool
from agent_service.pipeline import FALLBACK, Inbound

PHONE = "15550100001"


def msg(text, mid="wamid.1", phone=PHONE):
    return Inbound(message_id=mid, conversation_id=phone, text=text)


def test_tool_call_then_reply(make_service, store):
    svc = make_service(
        [
            tool("search_catalog", query="סיור"),
            reply("סיור מודרך לאורך החוף עולה 180 ₪ לאדם. לאיזה תאריך?"),
        ]
    )
    out = svc.handle(msg("כמה עולה סיור?"))
    assert out.status == "replied"
    assert out.language == "hebrew"
    assert out.tool_calls == ["search_catalog"]
    # the tool result was fed back to the model
    tool_msg = svc.llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and "coast-tour" in tool_msg["content"]
    turn = next(store.turns(PHONE))
    assert turn.outcome == "replied"
    assert turn.input_tokens == 200 and turn.cost_usd == pytest.approx(200 / 1e6 + 30 * 4 / 1e6)


def test_redelivered_message_is_answered_once(make_service):
    svc = make_service([reply("שלום! איך אפשר לעזור?")])
    first = svc.handle(msg("היי"))
    again = svc.handle(msg("היי"))  # same message id: provider redelivery
    assert first.status == "replied"
    assert again.status == "duplicate" and again.reply == first.reply
    assert len(svc.llm.calls) == 1  # no second LLM call


def test_same_conversation_is_serialised(make_service, store):
    order = []

    def slow(messages):
        order.append(("start", messages[-1]["content"]))
        time.sleep(0.1)
        order.append(("end", messages[-1]["content"]))
        return reply("קיבלתי, רגע בודק.")

    svc = make_service([slow, slow])
    t1 = threading.Thread(target=svc.handle, args=(msg("הודעה א", "m1"),))
    t2 = threading.Thread(target=svc.handle, args=(msg("הודעה ב", "m2"),))
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join()
    t2.join()
    assert [o[0] for o in order] == ["start", "end", "start", "end"]
    # the second turn saw the first turn's reply in its history
    assert svc.llm.calls[1][-2] == {"role": "assistant", "content": "קיבלתי, רגע בודק."}


def test_memory_reaches_the_prompt(make_service, store):
    store.save_fact(PHONE, "name", "דנה")
    store.save_fact(PHONE, "group_size", "4")
    svc = make_service([reply("היי דנה! במה אפשר לעזור?")])
    svc.handle(msg("שוב אני"))
    system = svc.llm.calls[0][0]["content"]
    assert "- name: דנה" in system and "- group_size: 4" in system
    assert "Reply in Hebrew only" in system


def test_bad_reply_is_rewritten_once(make_service):
    svc = make_service(
        [
            reply("Sure! A city bike is 60 ILS per day."),  # English to a Hebrew customer
            reply("אופני עיר ליום עולים 60 ₪. לאיזה תאריך?"),
        ]
    )
    out = svc.handle(msg("כמה עולה אופניים ליום?"))
    assert out.status == "replied" and "60 ₪" in out.reply
    note = svc.llm.calls[1][-1]["content"]
    assert "not written in the customer's language" in note


def test_two_bad_replies_fall_back_and_hand_off(make_service, store):
    svc = make_service([reply("As an AI language model..."), reply("I am an AI language model")])
    out = svc.handle(msg("מה שעות הפתיחה?"))
    assert out.status == "fallback_handoff"
    assert out.reply == FALLBACK["hebrew"]
    assert out.handed_off
    conv = store.get_or_create_conversation(PHONE)
    assert conv.state == "human" and conv.handoff_reason.startswith("fallback_handoff")


def test_bot_stays_silent_while_a_human_owns_the_conversation(make_service, store):
    store.update_conversation(PHONE, state="human")
    svc = make_service([])
    out = svc.handle(msg("עדכון: נהיה 5 אנשים"))
    assert out.status == "human_active" and out.reply is None
    assert svc.llm.calls == []
    assert store.recent_messages(PHONE, 5)[-1].content == "עדכון: נהיה 5 אנשים"  # still logged


def test_handoff_tool(make_service, store):
    svc = make_service(
        [
            tool("handoff_to_human", reason="refund request"),
            reply("אני מעביר אותך לנציג, הוא יענה כאן בהקדם."),
        ]
    )
    out = svc.handle(msg("אני רוצה החזר כספי"))
    assert out.status == "replied" and out.handed_off
    assert store.get_or_create_conversation(PHONE).state == "human"


def test_llm_outage_hands_off_politely(make_service, store):
    svc = make_service([LLMError("503 after retries")])
    out = svc.handle(msg("مرحبا، كم سعر الجولة؟"))
    assert out.status == "llm_error_handoff"
    assert out.reply == FALLBACK["arabic"]
    assert store.get_or_create_conversation(PHONE).state == "human"


def test_crash_releases_the_claim_so_the_retry_is_processed(make_service):
    svc = make_service([RuntimeError("bug"), reply("שלום! איך אפשר לעזור?")])
    with pytest.raises(RuntimeError):
        svc.handle(msg("היי"))
    out = svc.handle(msg("היי"))  # provider retries the same message id
    assert out.status == "replied"


def test_language_follows_the_customer(make_service, store):
    svc = make_service([reply("مرحبا! كيف يمكنني المساعدة؟"), reply("Hi! How can I help?")])
    assert svc.handle(msg("مرحبا", "m1")).language == "arabic"
    assert svc.handle(msg("hello", "m2")).language == "english"
    assert store.get_or_create_conversation(PHONE).language == "english"


def test_tool_loop_is_bounded(make_service):
    loop = [tool("search_catalog", query="x")] * 10
    svc = make_service(loop)
    out = svc.handle(msg("שאלה"))
    assert out.status == "fallback_handoff"
    assert out.guard_reasons == ["no_reply"]


def test_empty_text_is_ignored(make_service):
    assert make_service([]).handle(msg("   ")).status == "ignored"
