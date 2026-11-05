from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _to_messages(prompt: str | list[dict[str, str]], answer: str | list[dict[str, str]]):
    p = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    a = answer if isinstance(answer, list) else [{"role": "assistant", "content": str(answer)}]
    return p, a


def _response_logprob(model, tokenizer, prompt, answer, device: str, max_length: int) -> float:
    prompt_msgs, answer_msgs = _to_messages(prompt, answer)
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full_ids = tokenizer.apply_chat_template(
        prompt_msgs + answer_msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    original_full_len = full_ids.shape[1]
    prompt_len_orig = prompt_ids.shape[1]
    if original_full_len > max_length:
        truncated = original_full_len - max_length
        full_ids = full_ids[:, -max_length:]
        prompt_len = max(prompt_len_orig - truncated, 0)
    else:
        prompt_len = min(prompt_len_orig, full_ids.shape[1] - 1)

    full_ids = full_ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=full_ids).logits[:, :-1, :]
        targets = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # token_lp index i scores target token i+1; start at prompt_len-1.
    start = max(prompt_len - 1, 0)
    return float(token_lp[:, start:].sum().item())


def evaluate_preference_accuracy(
    model_path: str,
    dataset,
    *,
    max_samples: int = 500,
    max_length: int = 2048,
    device: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    correct = 0
    margins: list[float] = []
    n = min(len(dataset), max_samples)
    for i in range(n):
        row = dataset[i]
        chosen_lp = _response_logprob(model, tokenizer, row["prompt"], row["chosen"], device, max_length)
        rejected_lp = _response_logprob(model, tokenizer, row["prompt"], row["rejected"], device, max_length)
        margin = chosen_lp - rejected_lp
        margins.append(margin)
        correct += int(margin > 0)

    return {
        "samples": n,
        "preference_accuracy": correct / max(n, 1),
        "mean_chosen_minus_rejected_logprob": sum(margins) / max(len(margins), 1),
    }
