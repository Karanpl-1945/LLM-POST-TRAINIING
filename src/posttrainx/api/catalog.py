from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProductCatalog:
    def __init__(self, path: str):
        self.products: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Small demo retriever. Replace with OpenSearch/Elasticsearch/vector search in production."""
        terms = {t.lower() for t in query.split() if len(t) > 2}
        scored = []
        for p in self.products:
            haystack = " ".join(str(v) for v in p.values()).lower()
            score = sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]
