from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from typing import Any

from .base import Claim, Conversation, Lead, Role, StoredMessage, TurnRecord

_CONVERSATION_FIELDS = {"state", "language", "handoff_reason"}


class PostgresStore:
    """Production store. psycopg is imported lazily so the package works without it."""

    def __init__(self, dsn: str, max_connections: int = 10) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=max_connections, kwargs={"autocommit": True}, open=True
        )

    def close(self) -> None:
        self._pool.close()

    def init(self) -> None:
        sql = resources.files("agent_service.store").joinpath("schema.sql").read_text("utf-8")
        with self._pool.connection() as conn:
            conn.execute(sql)

    @contextmanager
    def lock(self, conversation_id: str):
        # Session-level advisory lock, held on a dedicated connection for the whole turn.
        # Works across processes and machines, unlike an in-process lock.
        with self._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (conversation_id,))
            try:
                yield
            finally:
                conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (conversation_id,)
                )

    def _one(self, sql: str, params: tuple = ()) -> tuple | None:
        with self._pool.connection() as conn:
            return conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def claim_message(self, message_id: str, conversation_id: str) -> Claim:
        row = self._one(
            "INSERT INTO inbound_messages (message_id, conversation_id) VALUES (%s, %s) "
            "ON CONFLICT (message_id) DO NOTHING RETURNING message_id",
            (message_id, conversation_id),
        )
        if row:
            return Claim(is_new=True)
        existing = self._one(
            "SELECT reply FROM inbound_messages WHERE message_id = %s", (message_id,)
        )
        return Claim(is_new=False, reply=existing[0] if existing else None)

    def complete_message(self, message_id: str, reply: str | None) -> None:
        self._exec(
            "UPDATE inbound_messages SET reply = %s, completed_at = now() WHERE message_id = %s",
            (reply, message_id),
        )

    def release_message(self, message_id: str) -> None:
        self._exec("DELETE FROM inbound_messages WHERE message_id = %s", (message_id,))

    def get_or_create_conversation(self, conversation_id: str) -> Conversation:
        self._exec(
            "INSERT INTO conversations (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (conversation_id,),
        )
        row = self._one(
            "SELECT id, state, language, handoff_reason FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        assert row is not None
        return Conversation(id=row[0], state=row[1], language=row[2], handoff_reason=row[3])

    def update_conversation(self, conversation_id: str, **fields: Any) -> None:
        unknown = set(fields) - _CONVERSATION_FIELDS
        if unknown:
            raise ValueError(f"unknown conversation fields: {sorted(unknown)}")
        if not fields:
            return
        sets = ", ".join(f"{k} = %s" for k in fields)  # keys are whitelisted above
        self._exec(
            f"UPDATE conversations SET {sets}, updated_at = now() WHERE id = %s",
            (*fields.values(), conversation_id),
        )

    def append_message(self, conversation_id: str, role: Role, content: str) -> None:
        self._exec(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )

    def recent_messages(self, conversation_id: str, limit: int) -> list[StoredMessage]:
        rows = self._all(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT %s",
            (conversation_id, limit),
        )
        return [StoredMessage(role=r[0], content=r[1]) for r in reversed(rows)]

    def get_facts(self, conversation_id: str) -> dict[str, str]:
        rows = self._all(
            "SELECT key, value FROM customer_facts WHERE conversation_id = %s ORDER BY key",
            (conversation_id,),
        )
        return {k: v for k, v in rows}

    def save_fact(self, conversation_id: str, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO customer_facts (conversation_id, key, value) VALUES (%s, %s, %s) "
            "ON CONFLICT (conversation_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (conversation_id, key, value),
        )

    def upsert_lead(self, lead: Lead) -> None:
        self._exec(
            "INSERT INTO leads (conversation_id, name, interest, preferred_date, language, status) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (conversation_id) DO UPDATE SET name = EXCLUDED.name, "
            "interest = EXCLUDED.interest, preferred_date = EXCLUDED.preferred_date, "
            "language = EXCLUDED.language, updated_at = now()",
            (
                lead.conversation_id,
                lead.name,
                lead.interest,
                lead.preferred_date,
                lead.language,
                lead.status,
            ),
        )

    def get_lead(self, conversation_id: str) -> Lead | None:
        row = self._one(
            "SELECT conversation_id, name, interest, preferred_date, language, status "
            "FROM leads WHERE conversation_id = %s",
            (conversation_id,),
        )
        return Lead(*row) if row else None

    def record_turn(self, turn: TurnRecord) -> None:
        self._exec(
            "INSERT INTO turns (conversation_id, message_id, outcome, models, input_tokens, "
            "output_tokens, cost_usd, latency_ms, attempts, tool_calls, guard_reasons) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                turn.conversation_id,
                turn.message_id,
                turn.outcome,
                json.dumps(turn.models),
                turn.input_tokens,
                turn.output_tokens,
                turn.cost_usd,
                turn.latency_ms,
                turn.attempts,
                json.dumps(turn.tool_calls),
                json.dumps(turn.guard_reasons),
            ),
        )

    def turns(self, conversation_id: str) -> Iterator[TurnRecord]:
        rows = self._all(
            "SELECT conversation_id, message_id, outcome, models, input_tokens, output_tokens, "
            "cost_usd, latency_ms, attempts, tool_calls, guard_reasons "
            "FROM turns WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        for r in rows:
            yield TurnRecord(
                conversation_id=r[0],
                message_id=r[1],
                outcome=r[2],
                models=list(r[3]),
                input_tokens=r[4],
                output_tokens=r[5],
                cost_usd=float(r[6]),
                latency_ms=r[7],
                attempts=r[8],
                tool_calls=list(r[9]),
                guard_reasons=list(r[10]),
            )
