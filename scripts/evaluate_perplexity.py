#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from datasets import load_dataset

from posttrainx.evaluation.perplexity import evaluate_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args()
    ds = load_dataset("json", data_files=args.data, split="train")
    texts = [str(ds[i][args.text_field]) for i in range(min(len(ds), args.max_samples))]
    print(json.dumps(evaluate_perplexity(args.model, texts, max_length=args.max_length), indent=2))


if __name__ == "__main__":
    main()
