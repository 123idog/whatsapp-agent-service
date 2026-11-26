"""Storage contract. Two implementations: in-memory (tests, demo) and Postgres (production).

Everything the agent knows about a customer lives here, not in process memory, so a
restart or a deploy never wipes a conversation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ConversationState = Literal["bot", "human"]
Role = Literal["customer", "assistant"]


@dataclass
class Conversation:
    id: str
    state: ConversationState = "bot"
    language: str | None = None
    handoff_reason: str | None = None


@dataclass
class StoredMessage:
    role: Role
    content: str


@dataclass
class Claim:
    """Result of claiming an inbound message id.

    is_new=False means the channel redelivered a message we already handled:
    return the stored reply and do nothing else (no second LLM call, no second lead).
    """

    is_new: bool
    reply: str | None = None


@dataclass
class Lead:
    conversation_id: str
    name: str | None
    interest: str | None
    preferred_date: str | None
    language: str | None
    status: str = "new"


@dataclass
class TurnRecord:
    conversation_id: str
    message_id: str
    outcome: str
    models: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    attempts: int = 0
    tool_calls: list[str] = field(default_factory=list)
    guard_reasons: list[str] = field(default_factory=list)


class Store(Protocol):
    def init(self) -> None: ...

    def lock(self, conversation_id: str) -> AbstractContextManager[None]:
        """Serialise processing per conversation. Customers send three short messages
        in a row; without this, three LLM calls race and answer out of order."""
        ...

    def claim_message(self, message_id: str, conversation_id: str) -> Claim: ...
    def complete_message(self, message_id: str, reply: str | None) -> None: ...
    def release_message(self, message_id: str) -> None:
        """Undo a claim after a crash so the channel's redelivery is processed."""
        ...

    def get_or_create_conversation(self, conversation_id: str) -> Conversation: ...
    def update_conversation(self, conversation_id: str, **fields: Any) -> None: ...

    def append_message(self, conversation_id: str, role: Role, content: str) -> None: ...
    def recent_messages(self, conversation_id: str, limit: int) -> list[StoredMessage]: ...

    def get_facts(self, conversation_id: str) -> dict[str, str]: ...
    def save_fact(self, conversation_id: str, key: str, value: str) -> None: ...

    def upsert_lead(self, lead: Lead) -> None: ...
    def get_lead(self, conversation_id: str) -> Lead | None: ...

    def record_turn(self, turn: TurnRecord) -> None: ...
    def turns(self, conversation_id: str) -> Iterator[TurnRecord]: ...
