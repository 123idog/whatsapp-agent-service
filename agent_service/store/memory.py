from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .base import Claim, Conversation, Lead, Role, StoredMessage, TurnRecord


class MemoryStore:
    """In-process store for tests and the no-database demo. Not for production:
    it forgets everything on restart, which is exactly the failure this project avoids."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._inbound: dict[str, dict[str, Any]] = {}
        self._conversations: dict[str, Conversation] = {}
        self._messages: dict[str, list[StoredMessage]] = defaultdict(list)
        self._facts: dict[str, dict[str, str]] = defaultdict(dict)
        self._leads: dict[str, Lead] = {}
        self._turns: list[TurnRecord] = []

    def init(self) -> None:
        pass

    @contextmanager
    def lock(self, conversation_id: str):
        with self._guard:
            lk = self._locks[conversation_id]
        with lk:
            yield

    def claim_message(self, message_id: str, conversation_id: str) -> Claim:
        with self._guard:
            row = self._inbound.get(message_id)
            if row is not None:
                return Claim(is_new=False, reply=row["reply"])
            self._inbound[message_id] = {"conversation_id": conversation_id, "reply": None}
            return Claim(is_new=True)

    def complete_message(self, message_id: str, reply: str | None) -> None:
        with self._guard:
            self._inbound[message_id]["reply"] = reply

    def release_message(self, message_id: str) -> None:
        with self._guard:
            self._inbound.pop(message_id, None)

    def get_or_create_conversation(self, conversation_id: str) -> Conversation:
        with self._guard:
            conv = self._conversations.setdefault(conversation_id, Conversation(conversation_id))
            return replace(conv)

    def update_conversation(self, conversation_id: str, **fields: Any) -> None:
        with self._guard:
            conv = self._conversations.setdefault(conversation_id, Conversation(conversation_id))
            for k, v in fields.items():
                setattr(conv, k, v)

    def append_message(self, conversation_id: str, role: Role, content: str) -> None:
        with self._guard:
            self._messages[conversation_id].append(StoredMessage(role, content))

    def recent_messages(self, conversation_id: str, limit: int) -> list[StoredMessage]:
        with self._guard:
            return list(self._messages[conversation_id][-limit:])

    def get_facts(self, conversation_id: str) -> dict[str, str]:
        with self._guard:
            return dict(self._facts[conversation_id])

    def save_fact(self, conversation_id: str, key: str, value: str) -> None:
        with self._guard:
            self._facts[conversation_id][key] = value

    def upsert_lead(self, lead: Lead) -> None:
        with self._guard:
            self._leads[lead.conversation_id] = lead

    def get_lead(self, conversation_id: str) -> Lead | None:
        with self._guard:
            return self._leads.get(conversation_id)

    def record_turn(self, turn: TurnRecord) -> None:
        with self._guard:
            self._turns.append(turn)

    def turns(self, conversation_id: str) -> Iterator[TurnRecord]:
        with self._guard:
            items = [t for t in self._turns if t.conversation_id == conversation_id]
        return iter(items)
