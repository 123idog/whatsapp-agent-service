from .base import Claim, Conversation, Lead, Store, StoredMessage, TurnRecord
from .memory import MemoryStore


def make_store(database_url: str | None) -> Store:
    if database_url:
        from .postgres import PostgresStore

        return PostgresStore(database_url)
    return MemoryStore()


__all__ = [
    "Claim",
    "Conversation",
    "Lead",
    "MemoryStore",
    "StoredMessage",
    "Store",
    "TurnRecord",
    "make_store",
]
