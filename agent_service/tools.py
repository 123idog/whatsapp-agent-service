"""Tools the model can call.

Rules every tool follows:
- validates its own input and returns {"error": ...} instead of raising, so the model can
  recover ("that date is a Saturday, we're closed") instead of the turn crashing
- writes are idempotent (upserts), because a turn can be retried after a guard failure
- results carry "tool" so logs and the demo model can tell them apart
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .catalog import Catalog
from .store import Lead, Store
from .text import Language, clean_name

log = logging.getLogger(__name__)

FACT_KEYS = ["name", "group_size", "preferred_date", "interest", "notes"]


@dataclass
class ToolContext:
    store: Store
    catalog: Catalog
    conversation_id: str
    language: Language
    today: date
    handed_off: bool = False
    calls: list[str] = field(default_factory=list)


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "search_catalog",
        "Search products and services (rentals, tours, repairs). Use before quoting any "
        "price. Never invent a price or a product that this tool did not return.",
        {
            "query": {
                "type": "string",
                "description": "What the customer is looking for, in their words",
            }
        },
        ["query"],
    ),
    _fn(
        "check_availability",
        "Check whether an item can be booked on a date.",
        {
            "item_id": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        ["item_id", "date"],
    ),
    _fn(
        "save_customer_fact",
        "Remember a fact the customer told you, so you never ask for it again.",
        {
            "key": {"type": "string", "enum": FACT_KEYS},
            "value": {"type": "string"},
        },
        ["key", "value"],
    ),
    _fn(
        "create_lead",
        "Create or update the booking request for the sales team once you know what the "
        "customer wants. Safe to call more than once.",
        {
            "interest": {"type": "string", "description": "Item(s) and group, one line"},
            "preferred_date": {"type": "string", "description": "YYYY-MM-DD if known"},
        },
        ["interest"],
    ),
    _fn(
        "handoff_to_human",
        "Pass the conversation to a person. Use when the customer asks for one, complains, "
        "asks for a discount or refund, or asks something the catalog cannot answer.",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
]


def search_catalog(ctx: ToolContext, query: str) -> dict[str, Any]:
    items = ctx.catalog.search(query, ctx.language)
    return {"items": items, "count": len(items)}


def check_availability(ctx: ToolContext, item_id: str, date: str) -> dict[str, Any]:
    item = ctx.catalog.get(item_id)
    if item is None:
        return {"error": f"unknown item_id {item_id!r}; call search_catalog first"}
    try:
        day = _parse_date(date)
    except ValueError:
        return {"error": "date must be YYYY-MM-DD"}
    if day < ctx.today:
        return {"error": "that date is in the past"}
    if day.weekday() == 5:  # Saturday
        return {"available": False, "reason": "closed on Saturdays"}
    return {"available": item["stock"] > 0, "item_id": item_id, "date": day.isoformat()}


def save_customer_fact(ctx: ToolContext, key: str, value: str) -> dict[str, Any]:
    if key not in FACT_KEYS:
        return {"error": f"key must be one of {FACT_KEYS}"}
    value = (value or "").strip()[:200]
    if key == "name":
        cleaned = clean_name(value)
        if not cleaned:
            return {"error": "not a usable name; ask the customer how to address them"}
        value = cleaned
    if not value:
        return {"error": "empty value"}
    ctx.store.save_fact(ctx.conversation_id, key, value)
    return {"saved": key}


def create_lead(
    ctx: ToolContext, interest: str, preferred_date: str | None = None
) -> dict[str, Any]:
    if preferred_date:
        try:
            preferred_date = _parse_date(preferred_date).isoformat()
        except ValueError:
            return {"error": "preferred_date must be YYYY-MM-DD"}
    facts = ctx.store.get_facts(ctx.conversation_id)
    ctx.store.upsert_lead(
        Lead(
            conversation_id=ctx.conversation_id,
            name=facts.get("name"),
            interest=interest.strip()[:300],
            preferred_date=preferred_date or facts.get("preferred_date"),
            language=ctx.language,
        )
    )
    return {"lead": "saved"}


def handoff_to_human(ctx: ToolContext, reason: str) -> dict[str, Any]:
    ctx.store.update_conversation(ctx.conversation_id, state="human", handoff_reason=reason[:200])
    ctx.handed_off = True
    return {"handoff": "done", "note": "tell the customer a person will reply here"}


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "search_catalog": search_catalog,
    "check_availability": check_availability,
    "save_customer_fact": save_customer_fact,
    "create_lead": create_lead,
    "handoff_to_human": handoff_to_human,
}


def execute(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    """Run a tool and return its JSON result for the model. Never raises."""
    ctx.calls.append(name)
    fn = REGISTRY.get(name)
    if fn is None:
        result: dict[str, Any] = {"error": f"unknown tool {name!r}"}
    else:
        try:
            result = fn(ctx, **arguments)
        except TypeError as e:  # wrong / missing arguments from the model
            result = {"error": f"bad arguments: {e}"}
        except Exception:  # a tool bug must not kill the turn
            log.exception("tool failed", extra={"tool": name})
            result = {"error": "internal error, try another way or hand off"}
    return json.dumps({"tool": name, **result}, ensure_ascii=False)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip()[:10])
