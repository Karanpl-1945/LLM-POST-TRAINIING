from __future__ import annotations

from typing import Any

from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from posttrainx.data.formatting import prompt_chosen_rejected_to_conversational
from posttrainx.data.loading import load_any_dataset, split_train_eval
from posttrainx.training.common import build_lora_config, load_model, load_tokenizer
from posttrainx.utils import ensure_dir, seed_everything, write_json


def _has_conversational_dpo_columns(example: dict[str, Any]) -> bool:
    return isinstance(example.get("prompt"), list) and isinstance(example.get("chosen"), list)


def run_dpo(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    model_id = cfg["model"]["id"]
    output_dir = ensure_dir(cfg["training"]["output_dir"])

    dataset = load_any_dataset(cfg, cfg["dataset"].get("train_split", "train"))
    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"DPO dataset must contain {sorted(required)}; found {dataset.column_names}")
    if len(dataset) and not _has_conversational_dpo_columns(dataset[0]):
        dataset = dataset.map(prompt_chosen_rejected_to_conversational)
    train_ds, eval_ds = split_train_eval(dataset, float(cfg["dataset"].get("eval_fraction", 0.02)), seed)

    model = load_model(model_id, {**cfg["model"], **cfg["peft"]})
    sft_adapter = cfg["model"].get("sft_adapter")
    if sft_adapter:
        # Load the SFT adapter into the policy. DPO gets a fresh trainable adapter via peft_config.
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=False)
        # Merge the SFT delta into the current in-memory model before attaching DPO LoRA.
        # For QLoRA workflows, if your PEFT/bitsandbytes versions do not support merging a
        # quantized model, first materialize a BF16 merged SFT checkpoint with merge_adapter.py,
        # then point model.id to that checkpoint.
        if bool(cfg["model"].get("merge_sft_adapter_before_dpo", False)):
            model = model.merge_and_unload()

    tokenizer = load_tokenizer(model_id, bool(cfg["model"].get("trust_remote_code", False)))
    tokenizer.padding_side = "left"
    peft_config = build_lora_config(cfg["peft"])
    t = cfg["training"]

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(t.get("num_train_epochs", 1.0)),
        max_steps=int(t.get("max_steps", -1)),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 4)),
        learning_rate=float(t.get("learning_rate", 5e-6)),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 250)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(t.get("eval_steps", 250)),
        save_total_limit=int(t.get("save_total_limit", 2)),
        bf16=t.get("precision", "bf16") == "bf16",
        fp16=t.get("precision") == "fp16",
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        max_length=int(t.get("max_length", 2048)),
        beta=float(t.get("beta", 0.1)),
        loss_type=t.get("loss_type", "sigmoid"),
        precompute_ref_log_probs=bool(t.get("precompute_ref_log_probs", False)),
        dataset_num_proc=int(t.get("dataset_num_proc", 4)),
        report_to=t.get("report_to", "none"),
        skip_memory_metrics=False,
        include_num_input_tokens_seen=True,
        seed=seed,
    )

    # With PEFT, TRL can construct/reference the frozen policy state internally. This avoids
    # keeping an extra full trainable copy in common adapter-based DPO setups.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train(resume_from_checkpoint=t.get("resume_from_checkpoint") or None)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    metrics = dict(result.metrics)
    if eval_ds is not None:
        metrics.update(trainer.evaluate())
    if metrics.get("train_runtime") and metrics.get("num_input_tokens_seen"):
        metrics["train_tokens_per_second"] = metrics["num_input_tokens_seen"] / metrics["train_runtime"]
    write_json(output_dir / "train_metrics.json", metrics)
    return metrics
