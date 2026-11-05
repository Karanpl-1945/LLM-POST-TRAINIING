from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_oasst1_pairs(rows: list[dict[str, Any]], language: str | None = "en") -> list[dict[str, Any]]:
    """Build simple chosen/rejected pairs from OASST1-ranked sibling assistant replies.

    OASST1 is tree structured. This utility:
      1. indexes messages by message_id,
      2. groups assistant children by parent_id,
      3. keeps groups with at least two ranked responses,
      4. uses the lowest rank as chosen and highest rank as rejected,
      5. uses the parent user's text as the prompt.

    Validate the source schema and ranking semantics for the exact revision you use.
    """
    by_id = {r.get("message_id"): r for r in rows if r.get("message_id")}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("role") != "assistant":
            continue
        if language and row.get("lang") not in (None, language):
            continue
        if row.get("parent_id") and row.get("rank") is not None and row.get("text"):
            children[row["parent_id"]].append(row)

    pairs: list[dict[str, Any]] = []
    for parent_id, replies in children.items():
        if len(replies) < 2:
            continue
        parent = by_id.get(parent_id)
        if not parent or parent.get("role") not in {"prompter", "user"} or not parent.get("text"):
            continue
        ranked = sorted(replies, key=lambda x: float(x["rank"]))
        chosen, rejected = ranked[0], ranked[-1]
        if chosen["text"].strip() == rejected["text"].strip():
            continue
        pairs.append(
            {
                "prompt": parent["text"].strip(),
                "chosen": chosen["text"].strip(),
                "rejected": rejected["text"].strip(),
                "metadata": {
                    "parent_id": parent_id,
                    "chosen_rank": chosen["rank"],
                    "rejected_rank": rejected["rank"],
                },
            }
        )
    return pairs
