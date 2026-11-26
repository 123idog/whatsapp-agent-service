"""Product catalog and retrieval.

Retrieval is deliberately simple: IDF-weighted keyword matching that understands Hebrew
and Arabic attached prefixes (לאופניים, بالدراجة). For a catalog of tens of items this
is precise, explainable and free. Swap `search` for a vector index
when the catalog grows; the tool contract stays the same.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .text import Language, contains_word, normalize


@dataclass
class Catalog:
    business: dict[str, Any]
    items: list[dict[str, Any]]

    @classmethod
    def load(cls, path: str | None = None) -> Catalog:
        if path:
            raw = Path(path).read_text("utf-8")
        else:
            raw = resources.files("agent_service.data").joinpath("catalog.json").read_text("utf-8")
        data = json.loads(raw)
        return cls(business=data["business"], items=data["items"])

    def business_name(self, language: Language) -> str:
        return self.business["name"].get(language) or self.business["name"]["english"]

    def get(self, item_id: str) -> dict[str, Any] | None:
        return next((i for i in self.items if i["id"] == item_id), None)

    def __post_init__(self) -> None:
        # Token index with IDF weights: a word shared by many items ("bike") counts less
        # than a word only one or two items have ("electric").
        self._tokens: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for item in self.items:
            terms = list(item["keywords"]) + list(item["name"].values())
            # len > 2 drops connectors that would match everything (مع, של, of)
            tokens = {t for term in terms for t in normalize(term).split() if len(t) > 2}
            self._tokens[item["id"]] = tokens
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        n = len(self.items)
        self._idf = {t: math.log(1 + n / d) for t, d in df.items()}

    def search(self, query: str, language: Language, limit: int = 3) -> list[dict[str, Any]]:
        prefixes = language if language in ("hebrew", "arabic") else None
        matched = {t for t in self._idf if contains_word(query, t, prefixes=prefixes)}
        scored = []
        for item in self.items:
            score = sum(self._idf[t] for t in self._tokens[item["id"]] & matched)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda s: -s[0])
        return [self.localize(item, language) for _, item in scored[:limit]]

    @staticmethod
    def localize(item: dict[str, Any], language: Language) -> dict[str, Any]:
        def pick(field: str) -> str:
            return item[field].get(language) or item[field]["english"]

        return {
            "id": item["id"],
            "name": pick("name"),
            "price": item["price"],
            "currency": "ILS",
            "unit": pick("unit"),
            "description": pick("description"),
        }
