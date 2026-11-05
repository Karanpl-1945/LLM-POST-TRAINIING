from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def evaluate_perplexity(
    model_path: str,
    texts: Iterable[str],
    *,
    max_length: int = 2048,
    device: str | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        labels = batch["input_ids"].clone()
        out = model(**batch, labels=labels)
        # HF causal-LM loss is mean over non-masked target tokens.
        n_tokens = max(int(labels.numel()) - 1, 1)
        total_nll += float(out.loss) * n_tokens
        total_tokens += n_tokens

    mean_nll = total_nll / max(total_tokens, 1)
    return {"mean_nll": mean_nll, "perplexity": math.exp(min(mean_nll, 50.0)), "tokens": total_tokens}
