from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent_service.catalog import Catalog
from agent_service.config import Settings
from agent_service.llm import ScriptedLLM
from agent_service.pipeline import AgentService
from agent_service.store import MemoryStore

# A Tuesday, so Saturday-closure logic is predictable.
NOW = datetime(2026, 6, 2, 10, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))


@pytest.fixture
def settings() -> Settings:
    return Settings(price_input_per_m=1.0, price_output_per_m=4.0)


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.load()


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def make_service(store, catalog, settings):
    def _make(steps, **kwargs) -> AgentService:
        return AgentService(
            store, ScriptedLLM(steps), catalog, settings, clock=lambda: NOW, **kwargs
        )

    return _make


def postgres_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")
