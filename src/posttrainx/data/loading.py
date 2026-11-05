from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def load_any_dataset(cfg: dict[str, Any], split: str):
    """Load a Hugging Face dataset or local JSON/JSONL/Parquet data.

    Config examples:
      dataset:
        source: hf
        name: org/name
        subset: optional
        train_split: train

      dataset:
        source: local
        path: data/processed/sft.jsonl
    """
    ds_cfg = cfg["dataset"]
    source = ds_cfg.get("source", "hf")

    if source == "hf":
        name = ds_cfg["name"]
        subset = ds_cfg.get("subset")
        revision = ds_cfg.get("revision")
        return load_dataset(name, subset, split=split, revision=revision)

    if source == "local":
        path = Path(ds_cfg["path"])
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset format: {suffix}")

    raise ValueError(f"Unsupported dataset source: {source}")


def split_train_eval(dataset: Dataset, eval_fraction: float, seed: int):
    if eval_fraction <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]
