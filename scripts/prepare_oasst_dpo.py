#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from posttrainx.data.oasst_preferences import build_oasst1_pairs


def main():
    p = argparse.ArgumentParser(description="Convert OASST1 ranked sibling replies into DPO triples")
    p.add_argument("--dataset", default="OpenAssistant/oasst1")
    p.add_argument("--split", default="train")
    p.add_argument("--language", default="en")
    p.add_argument("--output", default="data/processed/dpo_preferences.jsonl")
    args = p.parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    pairs = build_oasst1_pairs([dict(x) for x in ds], language=args.language or None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs):,} preference pairs to {out}")


if __name__ == "__main__":
    main()
