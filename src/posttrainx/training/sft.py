from __future__ import annotations

from typing import Any

from trl import SFTConfig, SFTTrainer

from posttrainx.data.formatting import normalize_messages
from posttrainx.data.loading import load_any_dataset, split_train_eval
from posttrainx.training.common import build_lora_config, load_model, load_tokenizer
from posttrainx.utils import ensure_dir, seed_everything, write_json


def run_sft(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    model_id = cfg["model"]["id"]
    output_dir = ensure_dir(cfg["training"]["output_dir"])

    dataset = load_any_dataset(cfg, cfg["dataset"].get("train_split", "train"))
    msg_field = cfg["dataset"].get("messages_field", "messages")
    if msg_field != "messages" or "messages" not in dataset.column_names:
        dataset = dataset.map(lambda x: normalize_messages(x, msg_field))
    dataset = dataset.filter(lambda x: len(x["messages"]) >= 2)
    train_ds, eval_ds = split_train_eval(dataset, float(cfg["dataset"].get("eval_fraction", 0.01)), seed)

    model = load_model(model_id, {**cfg["model"], **cfg["peft"]})
    tokenizer = load_tokenizer(model_id, bool(cfg["model"].get("trust_remote_code", False)))
    peft_config = build_lora_config(cfg["peft"])

    t = cfg["training"]
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(t.get("num_train_epochs", 1.0)),
        max_steps=int(t.get("max_steps", -1)),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        weight_decay=float(t.get("weight_decay", 0.0)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 250)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(t.get("eval_steps", 250)),
        save_total_limit=int(t.get("save_total_limit", 2)),
        bf16=t.get("precision", "bf16") == "bf16",
        fp16=t.get("precision") == "fp16",
        tf32=bool(t.get("tf32", True)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        max_length=int(t.get("max_length", 2048)),
        packing=bool(t.get("packing", True)),
        packing_strategy=t.get("packing_strategy", "bfd"),
        assistant_only_loss=bool(t.get("assistant_only_loss", False)),
        dataset_num_proc=int(t.get("dataset_num_proc", 4)),
        report_to=t.get("report_to", "none"),
        skip_memory_metrics=False,
        include_num_input_tokens_seen=True,
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
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
