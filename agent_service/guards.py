"""Output guards: decide whether an LLM reply is safe to send to a customer.

A guard failure is not an exception. The pipeline gets the reasons back, asks the model
to rewrite once with those reasons, and if the second attempt also fails it sends a
safe fallback message and hands the conversation to a human. A broken reply never
reaches the customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import EMAIL_RE, URL_RE, Language, ScriptCounts, detect_script, normalize

# Phrases that only appear when the model talks about its own instructions or tools.
DEFAULT_BLOCKLIST = [
    "system prompt",
    "my instructions",
    "as an ai language model",
    "tool call",
    "function call",
    "handoff_to_human",
    "search_catalog",
    "create_lead",
    "save_customer_fact",
    "check_availability",
]


@dataclass
class GuardResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    script: ScriptCounts | None = None


def find_prompt_leaks(reply: str, phrases: list[str]) -> list[str]:
    padded = f" {normalize(reply)} "
    return [p for p in phrases if (n := normalize(p)) and f" {n} " in padded]


def check_reply(
    reply: str | None,
    language: Language,
    *,
    blocklist: list[str] | None = None,
    max_chars: int = 700,
    max_questions: int = 1,
) -> GuardResult:
    """Checks, in order:

    - empty            nothing to send
    - mojibake         replacement chars, runs of "???", CJK characters
    - template_residue unrendered placeholders: {{name}}, ${x}, [[x]]
    - wrong_script     no letters in the customer's script (Hebrew/Arabic customers)
    - latin_dominant   more Latin than Hebrew/Arabic letters: English leaked in
    - prompt_leak      internal phrases (instructions, tool names) in the reply
    - too_long         a WhatsApp message nobody reads
    - too_many_questions  ask one thing at a time
    """
    text = reply or ""
    if not text.strip():
        return GuardResult(False, ["empty"])

    reasons: list[str] = []
    script = detect_script(EMAIL_RE.sub(" ", URL_RE.sub(" ", text)))

    if "�" in text or re.search(r"\?{3,}", text) or script.cjk > 0:
        reasons.append("mojibake")
    if re.search(r"\{\{[^}]*\}\}|\$\{[^}]*\}|\[\[[^\]]*\]\]", text):
        reasons.append("template_residue")

    if language in ("hebrew", "arabic"):
        expected = getattr(script, language)
        if expected == 0:
            reasons.append("wrong_script")
        elif script.latin > expected:
            reasons.append("latin_dominant")
    elif script.latin == 0:
        reasons.append("wrong_script")

    if find_prompt_leaks(text, blocklist if blocklist is not None else DEFAULT_BLOCKLIST):
        reasons.append("prompt_leak")
    if len(text) > max_chars:
        reasons.append("too_long")
    # Hebrew/English "?" and Arabic "؟"
    if len(re.findall(r"[?؟]", text)) > max_questions:
        reasons.append("too_many_questions")

    return GuardResult(not reasons, reasons, script)
