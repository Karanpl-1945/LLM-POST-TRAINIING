from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(base_model: str, adapter_path: str, output_dir: str, dtype: str = "bf16") -> None:
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
