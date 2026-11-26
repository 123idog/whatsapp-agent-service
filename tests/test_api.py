import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from agent_service.app import create_app
from agent_service.catalog import Catalog
from agent_service.channels import LogSender
from agent_service.config import Settings
from agent_service.llm import DemoLLM
from agent_service.pipeline import AgentService
from agent_service.store import MemoryStore

from .conftest import NOW

SECRET = "app-secret"


@pytest.fixture
def setup():
    settings = Settings(
        api_key="test-key", whatsapp_verify_token="verify-me", whatsapp_app_secret=SECRET
    )
    service = AgentService(MemoryStore(), DemoLLM(), Catalog.load(), settings, clock=lambda: NOW)
    sender = LogSender()
    return TestClient(create_app(service, sender, settings)), service, sender


AUTH = {"Authorization": "Bearer test-key"}


def test_healthz(setup):
    c, _, _ = setup
    assert c.get("/healthz").json() == {"status": "ok"}


def test_messages_requires_api_key(setup):
    c, _, _ = setup
    body = {"message_id": "m1", "conversation_id": "15550100001", "text": "היי"}
    assert c.post("/v1/messages", json=body).status_code == 401


def test_demo_conversation_end_to_end(setup):
    c, _, _ = setup
    phone = "15550100001"
    r = c.post(
        "/v1/messages",
        headers=AUTH,
        json={"message_id": "m1", "conversation_id": phone, "text": "כמה עולה סיור מודרך?"},
    )
    out = r.json()
    assert out["status"] == "replied" and "180" in out["reply"]
    r = c.post(
        "/v1/messages",
        headers=AUTH,
        json={"message_id": "m2", "conversation_id": phone, "text": "אני רוצה להזמין ל-4.6"},
    )
    assert r.json()["tool_calls"] == ["create_lead"]
    state = c.get(f"/v1/conversations/{phone}", headers=AUTH).json()
    assert state["lead"]["interest"] == "אני רוצה להזמין ל-4.6"
    assert len(state["turns"]) == 2 and len(state["messages"]) == 4


def test_release_returns_conversation_to_bot(setup):
    c, service, _ = setup
    service.store.update_conversation("15550100002", state="human")
    assert c.post("/v1/conversations/15550100002/release", headers=AUTH).json() == {"state": "bot"}
    assert service.store.get_or_create_conversation("15550100002").state == "bot"


def test_whatsapp_verification(setup):
    c, _, _ = setup
    params = {"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "123"}
    assert c.get("/webhooks/whatsapp", params=params).text == "123"
    params["hub.verify_token"] = "wrong"
    assert c.get("/webhooks/whatsapp", params=params).status_code == 403


def wa_payload(mid="wamid.A", text="مرحبا، بدي جولة مع مرشد"):
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"wa_id": "15550100009", "profile": {"name": "🌴"}}],
                            "messages": [
                                {
                                    "from": "15550100009",
                                    "id": mid,
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                            "statuses": [],
                        }
                    }
                ]
            }
        ],
    }


def signed(body: bytes) -> dict:
    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"x-hub-signature-256": f"sha256={sig}", "content-type": "application/json"}


def test_whatsapp_webhook_rejects_bad_signature(setup):
    c, _, sender = setup
    body = json.dumps(wa_payload()).encode()
    r = c.post(
        "/webhooks/whatsapp",
        content=body,
        headers={"x-hub-signature-256": "sha256=deadbeef", "content-type": "application/json"},
    )
    assert r.status_code == 401 and sender.sent == []


def test_whatsapp_webhook_replies_once_despite_redelivery(setup):
    c, _, sender = setup
    body = json.dumps(wa_payload()).encode()
    assert c.post("/webhooks/whatsapp", content=body, headers=signed(body)).json() == {
        "accepted": 1
    }
    c.post("/webhooks/whatsapp", content=body, headers=signed(body))  # Meta redelivers
    assert len(sender.sent) == 1
    to, text = sender.sent[0]
    assert to == "15550100009" and "180" in text and "شيكل" in text


def test_status_callbacks_are_ignored(setup):
    c, _, sender = setup
    payload = {
        "entry": [{"changes": [{"value": {"statuses": [{"id": "wamid.A", "status": "read"}]}}]}]
    }
    body = json.dumps(payload).encode()
    assert c.post("/webhooks/whatsapp", content=body, headers=signed(body)).json() == {
        "accepted": 0
    }
    assert sender.sent == []
