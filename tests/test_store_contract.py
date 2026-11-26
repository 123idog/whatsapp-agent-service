"""The same contract runs against MemoryStore and, when TEST_DATABASE_URL is set, Postgres."""

import threading
import time
import uuid

import pytest

from agent_service.store import Lead, MemoryStore, TurnRecord

from .conftest import postgres_url


@pytest.fixture(params=["memory", "postgres"])
def any_store(request):
    if request.param == "memory":
        yield MemoryStore()
        return
    url = postgres_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from agent_service.store.postgres import PostgresStore

    s = PostgresStore(url)
    s.init()
    yield s
    s.close()


def cid() -> str:
    return f"test-{uuid.uuid4().hex[:10]}"


def test_claim_complete_and_redelivery(any_store):
    c, m = cid(), f"wamid.{uuid.uuid4().hex}"
    assert any_store.claim_message(m, c).is_new
    again = any_store.claim_message(m, c)
    assert not again.is_new and again.reply is None  # still processing
    any_store.complete_message(m, "hi")
    assert any_store.claim_message(m, c).reply == "hi"


def test_release_allows_reprocessing(any_store):
    c, m = cid(), f"wamid.{uuid.uuid4().hex}"
    any_store.claim_message(m, c)
    any_store.release_message(m)
    assert any_store.claim_message(m, c).is_new


def test_conversation_messages_and_facts(any_store):
    c = cid()
    conv = any_store.get_or_create_conversation(c)
    assert conv.state == "bot"
    any_store.update_conversation(c, state="human", language="arabic", handoff_reason="x")
    conv = any_store.get_or_create_conversation(c)
    assert (conv.state, conv.language, conv.handoff_reason) == ("human", "arabic", "x")

    for i in range(5):
        any_store.append_message(c, "customer" if i % 2 == 0 else "assistant", f"m{i}")
    recent = any_store.recent_messages(c, 3)
    assert [m.content for m in recent] == ["m2", "m3", "m4"]  # oldest first

    any_store.save_fact(c, "name", "דנה")
    any_store.save_fact(c, "name", "Dana")
    assert any_store.get_facts(c) == {"name": "Dana"}


def test_lead_upsert(any_store):
    c = cid()
    any_store.get_or_create_conversation(c)
    any_store.upsert_lead(Lead(c, "Dana", "tour", None, "hebrew"))
    any_store.upsert_lead(Lead(c, "Dana", "tour x3", "2026-06-04", "hebrew"))
    lead = any_store.get_lead(c)
    assert (lead.interest, lead.preferred_date) == ("tour x3", "2026-06-04")


def test_turns(any_store):
    c = cid()
    any_store.record_turn(
        TurnRecord(
            c,
            "m1",
            "replied",
            models=["a"],
            input_tokens=10,
            cost_usd=0.0012,
            tool_calls=["search_catalog"],
        )
    )
    t = next(iter(any_store.turns(c)))
    assert t.tool_calls == ["search_catalog"] and t.cost_usd == pytest.approx(0.0012)


def test_lock_serialises_one_conversation(any_store):
    c, events = cid(), []

    def worker(name):
        with any_store.lock(c):
            events.append(f"{name}-in")
            time.sleep(0.1)
            events.append(f"{name}-out")

    a = threading.Thread(target=worker, args=("a",))
    b = threading.Thread(target=worker, args=("b",))
    a.start()
    time.sleep(0.03)
    b.start()
    a.join()
    b.join()
    assert events == ["a-in", "a-out", "b-in", "b-out"]
