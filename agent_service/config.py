from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    llm_provider: str = "demo"  # demo | openai_compat
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str = ""
    llm_timeout: float = 30.0
    price_input_per_m: float = 0.0  # USD per 1M input tokens, for cost tracking
    price_output_per_m: float = 0.0

    database_url: str = ""  # empty = in-memory store
    catalog_path: str = ""
    default_language: str = "hebrew"
    timezone: str = "Asia/Jerusalem"
    history_messages: int = 20
    max_reply_chars: int = 700

    api_key: str = ""  # bearer token for /v1 endpoints; empty = open (local dev only)
    whatsapp_verify_token: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_provider=_env("LLM_PROVIDER", "demo"),
            llm_base_url=_env("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=_env("LLM_API_KEY"),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            llm_fallback_model=_env("LLM_FALLBACK_MODEL"),
            llm_timeout=float(_env("LLM_TIMEOUT", "30")),
            price_input_per_m=float(_env("PRICE_INPUT_PER_M", "0")),
            price_output_per_m=float(_env("PRICE_OUTPUT_PER_M", "0")),
            database_url=_env("DATABASE_URL"),
            catalog_path=_env("CATALOG_PATH"),
            default_language=_env("DEFAULT_LANGUAGE", "hebrew"),
            timezone=_env("BUSINESS_TIMEZONE", "Asia/Jerusalem"),
            history_messages=int(_env("HISTORY_MESSAGES", "20")),
            max_reply_chars=int(_env("MAX_REPLY_CHARS", "700")),
            api_key=_env("API_KEY"),
            whatsapp_verify_token=_env("WHATSAPP_VERIFY_TOKEN"),
            whatsapp_app_secret=_env("WHATSAPP_APP_SECRET"),
            whatsapp_token=_env("WHATSAPP_TOKEN"),
            whatsapp_phone_number_id=_env("WHATSAPP_PHONE_NUMBER_ID"),
        )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens * self.price_input_per_m / 1e6
            + output_tokens * self.price_output_per_m / 1e6,
            6,
        )
