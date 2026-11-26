"""WhatsApp Cloud API: inbound webhook parsing, signature check, outbound sender."""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Protocol

import httpx

from .pipeline import Inbound

log = logging.getLogger(__name__)

GRAPH_URL = "https://graph.facebook.com/v21.0"


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """Meta signs every webhook with HMAC-SHA256 of the raw body using the app secret."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def parse_webhook(payload: dict[str, Any]) -> list[Inbound]:
    """Extract text messages. Status updates (sent/delivered/read) and media are skipped:
    a status callback must never trigger a reply."""
    out: list[Inbound] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            names = {
                c.get("wa_id"): (c.get("profile") or {}).get("name")
                for c in value.get("contacts") or []
            }
            for m in value.get("messages") or []:
                if m.get("type") != "text":
                    continue
                out.append(
                    Inbound(
                        message_id=m["id"],
                        conversation_id=m["from"],
                        text=(m.get("text") or {}).get("body", ""),
                        profile_name=names.get(m["from"]),
                    )
                )
    return out


class Sender(Protocol):
    def send(self, to: str, text: str) -> None: ...


class LogSender:
    """Default for local runs: prints instead of sending."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, text: str) -> None:
        self.sent.append((to, text))
        log.info("outbound (not sent, LogSender)", extra={"to": to, "text": text})


class WhatsAppCloudSender:
    def __init__(
        self, token: str, phone_number_id: str, transport: httpx.BaseTransport | None = None
    ):
        self._client = httpx.Client(
            base_url=GRAPH_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
            transport=transport,
        )
        self._path = f"/{phone_number_id}/messages"

    def send(self, to: str, text: str) -> None:
        body = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
        last: str = ""
        for _ in range(3):
            try:
                resp = self._client.post(self._path, json=body)
            except httpx.TransportError as e:
                last = str(e)
                continue
            if resp.status_code < 300:
                return
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code < 500 and resp.status_code != 429:
                break
        # A 200 from the send API is not proof of delivery either; delivery arrives later
        # as a status webhook. Here we only know the send itself failed.
        raise RuntimeError(f"WhatsApp send failed: {last}")
