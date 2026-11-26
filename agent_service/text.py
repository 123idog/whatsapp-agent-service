"""Script detection and normalisation for Hebrew, Arabic and Latin text.

Python port of the core of https://github.com/123idog/hebrew-arabic-guards.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Language = Literal["hebrew", "arabic", "english"]

_HEBREW = re.compile(r"[א-תװ-ײיִ-ﭏ]")
_ARABIC = re.compile(r"[ؠ-يٱ-ۓۺ-ۼݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

# Hebrew points/cantillation, Arabic harakat and tatweel.
_HEBREW_MARKS = re.compile(r"[֑-ׇ]")
_ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


@dataclass(frozen=True)
class ScriptCounts:
    hebrew: int
    arabic: int
    latin: int
    cjk: int

    @property
    def dominant(self) -> str:
        best, best_n = "none", 0
        for name in ("hebrew", "arabic", "latin", "cjk"):
            n = getattr(self, name)
            if n > best_n:
                best, best_n = name, n
        return best


def detect_script(text: str | None) -> ScriptCounts:
    s = text or ""
    return ScriptCounts(
        hebrew=len(_HEBREW.findall(s)),
        arabic=len(_ARABIC.findall(s)),
        latin=len(_LATIN.findall(s)),
        cjk=len(_CJK.findall(s)),
    )


def detect_language(text: str | None, default: Language = "hebrew") -> Language:
    """Language the customer wrote in. URLs and emails don't count."""
    stripped = EMAIL_RE.sub(" ", URL_RE.sub(" ", text or ""))
    dominant = detect_script(stripped).dominant
    if dominant == "hebrew":
        return "hebrew"
    if dominant == "arabic":
        return "arabic"
    if dominant == "latin":
        return "english"
    return default


def normalize(text: str | None) -> str:
    """Strip diacritics, unify Arabic alef/yeh variants, lowercase, drop punctuation."""
    s = unicodedata.normalize("NFC", text or "")
    s = _HEBREW_MARKS.sub("", s)
    s = _ARABIC_MARKS.sub("", s)
    s = re.sub("[آأإٱ]", "ا", s)  # آ أ إ ٱ -> ا
    s = s.replace("ى", "ي")  # ى -> ي
    s = s.lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


# Optional attached prefixes: Hebrew ו ה ב ל מ ש כ, Arabic و/ف + ب/ل/ك + ال.
_PREFIX = {
    "hebrew": "(?:[והבלמשכ]{1,4})?",
    # و/ف, then ب/ل/ك, then ال; plus لل (ل + ال, as in للعائلة)
    "arabic": "(?:[وف])?(?:لل|(?:[بلك])?(?:ال)?)",
}


def contains_word(text: str, word: str, prefixes: str | None = None) -> bool:
    """Whole-word match that works for Hebrew and Arabic (Python's \\b is fine; JS's isn't),
    with optional attached prefixes, after normalisation."""
    t, w = normalize(text), normalize(word)
    if not w:
        return False
    prefix = _PREFIX.get(prefixes or "", "")
    return re.search(rf"(?<!\w){prefix}{re.escape(w)}(?!\w)", t) is not None


def clean_name(name: str | None, max_words: int = 4, max_length: int = 40) -> str | None:
    """Keep letters (any script), marks, spaces, apostrophes, hyphens. None if unusable."""
    s = unicodedata.normalize("NFC", name or "").replace("ـ", "")
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M") or ch.isspace() or ch in "'׳’-":
            out.append(ch)
        else:
            out.append(" ")
    cleaned = re.sub(r"\s+", " ", "".join(out))
    cleaned = re.sub(r"(^|\s)['׳’-]+|['׳’-]+(?=\s|$)", r"\1", cleaned).strip()
    if not cleaned or not any(unicodedata.category(c).startswith("L") for c in cleaned):
        return None
    if len(cleaned.split(" ")) > max_words or len(cleaned) > max_length:
        return None
    return cleaned
