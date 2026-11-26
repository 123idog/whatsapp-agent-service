"""One inbound message in, at most one reply out.

claim message id (dedup) ─► conversation lock ─► human active? stay silent
      │
      ▼
memory: facts + recent messages ─► agent loop (tools) ─► output guards
                                                           │ fail
                                                           ▼
                                          rewrite once with the reasons
                                                           │ fail again
                                                           ▼
                                          safe fallback + hand off to a person
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from string import Template
from typing import Any
from zoneinfo import ZoneInfo

from .agent import run_agent
from .catalog import Catalog
from .config import Settings
from .guards import check_reply
from .llm import LLM, LLMError
from .store import Store, TurnRecord
from .text import Language, detect_language
from .tools import ToolContext

log = logging.getLogger("agent_service.turn")

LANGUAGE_NAMES = {"hebrew": "Hebrew", "arabic": "Arabic", "english": "English"}

FALLBACK = {
    "hebrew": "תודה על ההודעה! נציג מהצוות ימשיך איתך כאן בהקדם.",
    "arabic": "شكرا على رسالتك! سيتابع معك أحد أعضاء الفريق هنا قريبا.",
    "english": "Thanks for your message! A team member will continue with you here shortly.",
}

GUARD_HINTS = {
    "empty": "it was empty",
    "mojibake": "it contained broken characters",
    "template_residue": "it contained an unfilled placeholder",
    "wrong_script": "it was not written in the customer's language",
    "latin_dominant": "it was mostly English instead of the customer's language",
    "prompt_leak": "it mentioned internal instructions or tool names",
    "too_long": "it was too long for WhatsApp",
    "too_many_questions": "it asked more than one question",
}


@dataclass
class Inbound:
    message_id: str
    conversation_id: str  # the customer's phone number in WhatsApp
    text: str
    profile_name: str | None = None


@dataclass
class Outcome:
    status: (
        str  # replied | duplicate | human_active | fallback_handoff | llm_error_handoff | ignored
    )
    reply: str | None
    conversation_id: str
    language: str | None = None
    handed_off: bool = False
    guard_reasons: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


class AgentService:
    def __init__(
        self,
        store: Store,
        llm: LLM,
        catalog: Catalog,
        settings: Settings,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.catalog = catalog
        self.settings = settings
        tz = ZoneInfo(settings.timezone)
        self._clock = clock or (lambda: datetime.now(tz))
        self._prompt = Template(
            resources.files("agent_service.prompts").joinpath("system.md").read_text("utf-8")
        )

    def handle(self, msg: Inbound) -> Outcome:
        if not msg.text.strip():
            return Outcome("ignored", None, msg.conversation_id)

        with self.store.lock(msg.conversation_id):
            claim = self.store.claim_message(msg.message_id, msg.conversation_id)
            if not claim.is_new:
                return Outcome("duplicate", claim.reply, msg.conversation_id)
            try:
                outcome = self._process(msg)
            except Exception:
                # Undo the claim so the channel's retry is processed instead of swallowed.
                self.store.release_message(msg.message_id)
                raise
            self.store.complete_message(msg.message_id, outcome.reply)
            return outcome

    def _process(self, msg: Inbound) -> Outcome:
        started = time.monotonic()
        conv = self.store.get_or_create_conversation(msg.conversation_id)
        language: Language = detect_language(
            msg.text,
            default=conv.language or self.settings.default_language,  # type: ignore[arg-type]
        )
        if language != conv.language:
            self.store.update_conversation(conv.id, language=language)
        self.store.append_message(conv.id, "customer", msg.text)

        if conv.state == "human":
            # A person owns this conversation. The bot records the message and stays quiet.
            return Outcome("human_active", None, conv.id, language)

        now = self._clock()
        ctx = ToolContext(
            store=self.store,
            catalog=self.catalog,
            conversation_id=conv.id,
            language=language,
            today=now.date(),
        )
        messages = self._build_messages(conv.id, language, now)
        turn = TurnRecord(conversation_id=conv.id, message_id=msg.message_id, outcome="")

        reply: str | None = None
        reasons: list[str] = []
        try:
            for attempt in (1, 2):
                turn.attempts = attempt
                run = run_agent(self.llm, messages, ctx)
                turn.input_tokens += run.input_tokens
                turn.output_tokens += run.output_tokens
                turn.models += [m for m in run.models if m not in turn.models]

                guard = check_reply(run.text, language, max_chars=self.settings.max_reply_chars)
                if run.text is None:
                    guard.ok, guard.reasons = False, ["no_reply"]
                if guard.ok:
                    reply, reasons = run.text, []
                    break
                reasons = guard.reasons
                messages += self._rewrite_request(run.text, guard.reasons, language)
            status = "replied" if reply else "fallback_handoff"
        except LLMError as e:
            log.error("llm unavailable", extra={"conversation_id": conv.id, "error": str(e)})
            status = "llm_error_handoff"
            reasons = ["llm_error"]

        if reply is None:
            reply = FALLBACK[language]
            if not ctx.handed_off:
                self.store.update_conversation(
                    conv.id, state="human", handoff_reason=f"{status}: {','.join(reasons)}"
                )
                ctx.handed_off = True

        self.store.append_message(conv.id, "assistant", reply)

        turn.outcome = status
        turn.tool_calls = list(ctx.calls)
        turn.guard_reasons = reasons
        turn.latency_ms = int((time.monotonic() - started) * 1000)
        turn.cost_usd = self.settings.cost_usd(turn.input_tokens, turn.output_tokens)
        self.store.record_turn(turn)
        log.info("turn", extra={"turn": turn.__dict__, "language": language})

        return Outcome(
            status=status,
            reply=reply,
            conversation_id=conv.id,
            language=language,
            handed_off=ctx.handed_off,
            guard_reasons=reasons,
            tool_calls=list(ctx.calls),
        )

    def _build_messages(
        self, conversation_id: str, language: Language, now: datetime
    ) -> list[dict[str, Any]]:
        facts = self.store.get_facts(conversation_id)
        facts_text = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "- nothing yet"
        system = self._prompt.safe_substitute(
            business_name=self.catalog.business_name(language),
            business_about=self.catalog.business.get("about", ""),
            hours=self.catalog.business.get("hours", ""),
            today=now.date().isoformat(),
            weekday=now.strftime("%A"),
            language_name=LANGUAGE_NAMES[language],
            facts=facts_text,
        )
        history = self.store.recent_messages(conversation_id, self.settings.history_messages)
        return [{"role": "system", "content": system}] + [
            {"role": "user" if m.role == "customer" else "assistant", "content": m.content}
            for m in history
        ]

    @staticmethod
    def _rewrite_request(
        text: str | None, reasons: list[str], language: Language
    ) -> list[dict[str, Any]]:
        why = "; ".join(GUARD_HINTS.get(r, r) for r in reasons)
        note = (
            f"[internal note, not from the customer] Your previous reply was not sent because {why}. "
            f"Write the reply again, in {LANGUAGE_NAMES[language]}, short, one question at most."
        )
        out: list[dict[str, Any]] = []
        if text:
            out.append({"role": "assistant", "content": text})
        out.append({"role": "user", "content": note})
        return out
