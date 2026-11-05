from __future__ import annotations

from typing import Any


def normalize_messages(example: dict[str, Any], field: str = "messages") -> dict[str, Any]:
    """Normalize common role/content aliases into TRL conversational format."""
    raw = example[field]
    messages = []
    role_map = {"human": "user", "prompter": "user", "gpt": "assistant", "bot": "assistant"}
    for item in raw:
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value") or item.get("text")
        role = role_map.get(role, role)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role: {role}")
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content.strip()})
    return {"messages": messages}


def prompt_chosen_rejected_to_conversational(example: dict[str, Any]) -> dict[str, Any]:
    """Convert string preference triples to conversational DPO format."""
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    if isinstance(prompt, list):
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return {
        "prompt": [{"role": "user", "content": str(prompt)}],
        "chosen": [{"role": "assistant", "content": str(chosen)}],
        "rejected": [{"role": "assistant", "content": str(rejected)}],
    }
