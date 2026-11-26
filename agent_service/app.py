from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .catalog import Catalog
from .channels import LogSender, Sender, WhatsAppCloudSender, parse_webhook, verify_signature
from .config import Settings
from .llm import LLM, DemoLLM, OpenAICompatLLM
from .pipeline import AgentService, Inbound
from .store import make_store

log = logging.getLogger("agent_service")


class MessageIn(BaseModel):
    message_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=100)
    text: str = Field(max_length=4000)
    profile_name: str | None = None


class JsonFormatter(logging.Formatter):
    _skip = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update({k: v for k, v in record.__dict__.items() if k not in self._skip})
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai_compat":
        if not settings.llm_api_key:
            raise RuntimeError("LLM_PROVIDER=openai_compat needs LLM_API_KEY")
        return OpenAICompatLLM(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            fallback_model=settings.llm_fallback_model or None,
            timeout=settings.llm_timeout,
        )
    return DemoLLM()


def build_service(settings: Settings) -> AgentService:
    store = make_store(settings.database_url)
    store.init()
    return AgentService(
        store, build_llm(settings), Catalog.load(settings.catalog_path or None), settings
    )


def build_sender(settings: Settings) -> Sender:
    if settings.whatsapp_token and settings.whatsapp_phone_number_id:
        return WhatsAppCloudSender(settings.whatsapp_token, settings.whatsapp_phone_number_id)
    return LogSender()


def create_app(
    service: AgentService | None = None,
    sender: Sender | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or (service.settings if service else Settings.from_env())
    service = service or build_service(settings)
    sender = sender or build_sender(settings)
    app = FastAPI(title="whatsapp-agent-service", version="0.1.0")
    app.state.service = service
    app.state.sender = sender

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages", dependencies=[Depends(require_api_key)])
    def post_message(body: MessageIn) -> dict[str, Any]:
        """Synchronous entry point for any channel: send a message, get the reply back."""
        outcome = service.handle(Inbound(**body.model_dump()))
        return asdict(outcome)

    @app.get("/v1/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        store = service.store
        conv = store.get_or_create_conversation(conversation_id)
        lead = store.get_lead(conversation_id)
        return {
            "conversation": asdict(conv),
            "facts": store.get_facts(conversation_id),
            "lead": asdict(lead) if lead else None,
            "messages": [asdict(m) for m in store.recent_messages(conversation_id, 50)],
            "turns": [asdict(t) for t in store.turns(conversation_id)],
        }

    @app.post(
        "/v1/conversations/{conversation_id}/release", dependencies=[Depends(require_api_key)]
    )
    def release(conversation_id: str) -> dict[str, str]:
        """A person hands the conversation back to the bot."""
        service.store.update_conversation(conversation_id, state="bot", handoff_reason=None)
        return {"state": "bot"}

    @app.get("/webhooks/whatsapp")
    def whatsapp_verify(
        mode: str = Query(alias="hub.mode", default=""),
        token: str = Query(alias="hub.verify_token", default=""),
        challenge: str = Query(alias="hub.challenge", default=""),
    ) -> PlainTextResponse:
        if (
            mode == "subscribe"
            and settings.whatsapp_verify_token
            and token == settings.whatsapp_verify_token
        ):
            return PlainTextResponse(challenge)
        raise HTTPException(status_code=403, detail="verification failed")

    @app.post("/webhooks/whatsapp")
    async def whatsapp_webhook(request: Request, background: BackgroundTasks) -> dict[str, int]:
        raw = await request.body()
        if settings.whatsapp_app_secret:
            if not verify_signature(
                settings.whatsapp_app_secret, raw, request.headers.get("x-hub-signature-256")
            ):
                raise HTTPException(status_code=401, detail="bad signature")
        else:
            log.warning("WHATSAPP_APP_SECRET not set: webhook signature NOT verified")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="invalid JSON") from e
        messages = parse_webhook(payload)
        for msg in messages:
            background.add_task(_handle_and_send, service, sender, msg)
        # Acknowledge fast: Meta retries slow webhooks, and the dedup absorbs the retries.
        return {"accepted": len(messages)}

    return app


def _handle_and_send(service: AgentService, sender: Sender, msg: Inbound) -> None:
    try:
        outcome = service.handle(msg)
        if outcome.reply and outcome.status != "duplicate":
            sender.send(msg.conversation_id, outcome.reply)
    except Exception:
        log.exception("failed to handle inbound message", extra={"message_id": msg.message_id})
