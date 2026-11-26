import json
from datetime import date

from agent_service.tools import ToolContext, execute


def ctx(store, catalog, language="hebrew"):
    return ToolContext(
        store=store,
        catalog=catalog,
        conversation_id="15550100001",
        language=language,
        today=date(2026, 6, 2),
    )


def call(c, name, **args):
    return json.loads(execute(c, name, args))


def test_search_hebrew_with_prefix(catalog):
    ids = [i["id"] for i in catalog.search("כמה עולה להשכיר אופניים חשמליים?", "hebrew")]
    assert ids[0] in ("ebike-day", "ebike-hour")


def test_search_arabic_localises(catalog):
    items = catalog.search("بدي جولة مع مرشد", "arabic")
    assert items[0]["id"] == "coast-tour"
    assert items[0]["name"] == "جولة مع مرشد على طول الشاطئ"


def test_search_arabic_lil_prefix(catalog):
    # للعائلة = ل + ال + عائلة; connector words like مع must not decide the ranking
    assert catalog.search("بدي جولة للعائلة مع الاطفال", "arabic")[0]["id"] == "family-ride"


def test_specific_words_outweigh_shared_ones(catalog):
    assert catalog.search("כמה עולה טיפול לאופניים", "hebrew")[0]["id"] == "tune-up"
    assert catalog.search("I need to fix a flat tire", "english")[0]["id"] == "puncture-repair"


def test_search_nothing(catalog):
    assert catalog.search("pizza delivery", "english") == []


def test_availability_rules(store, catalog):
    c = ctx(store, catalog)
    assert (
        call(c, "check_availability", item_id="coast-tour", date="2026-06-04")["available"] is True
    )
    sat = call(c, "check_availability", item_id="coast-tour", date="2026-06-06")
    assert sat["available"] is False and "Saturday" in sat["reason"]
    assert "past" in call(c, "check_availability", item_id="coast-tour", date="2026-05-01")["error"]
    assert (
        "unknown item_id" in call(c, "check_availability", item_id="x", date="2026-06-04")["error"]
    )
    assert (
        "YYYY-MM-DD"
        in call(c, "check_availability", item_id="coast-tour", date="next week")["error"]
    )


def test_save_fact_cleans_name_and_rejects_junk(store, catalog):
    c = ctx(store, catalog)
    assert call(c, "save_customer_fact", key="name", value="דנה 🌸")["saved"] == "name"
    assert store.get_facts(c.conversation_id)["name"] == "דנה"
    assert "error" in call(c, "save_customer_fact", key="name", value="🔥🔥")
    assert "error" in call(c, "save_customer_fact", key="credit_card", value="4111")


def test_create_lead_is_idempotent_and_uses_memory(store, catalog):
    c = ctx(store, catalog)
    call(c, "save_customer_fact", key="name", value="Dana")
    call(c, "create_lead", interest="coast tour x2")
    call(c, "create_lead", interest="coast tour x3", preferred_date="2026-06-04")
    lead = store.get_lead(c.conversation_id)
    assert (lead.name, lead.interest, lead.preferred_date) == (
        "Dana",
        "coast tour x3",
        "2026-06-04",
    )


def test_handoff_sets_state(store, catalog):
    c = ctx(store, catalog)
    call(c, "handoff_to_human", reason="wants a refund")
    conv = store.get_or_create_conversation(c.conversation_id)
    assert conv.state == "human" and c.handed_off


def test_bad_tool_input_never_raises(store, catalog):
    c = ctx(store, catalog)
    assert "unknown tool" in call(c, "delete_everything")["error"]
    assert "bad arguments" in call(c, "search_catalog", q="wrong arg name")["error"]
