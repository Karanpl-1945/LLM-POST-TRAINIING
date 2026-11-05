from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def compute_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_quantization_config(cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if cfg.get("method", "lora").lower() != "qlora":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype(cfg.get("compute_dtype", "bf16")),
        bnb_4bit_use_double_quant=bool(cfg.get("double_quant", True)),
    )


def load_tokenizer(model_id: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, cfg: dict[str, Any]):
    dtype = compute_dtype(cfg.get("dtype", "bf16"))
    qcfg = build_quantization_config(cfg)
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        "torch_dtype": dtype,
    }
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    if qcfg is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    return model


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    targets = cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=targets,
        bias=cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
