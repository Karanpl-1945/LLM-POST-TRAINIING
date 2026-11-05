#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from datasets import load_dataset

from posttrainx.evaluation.preference import evaluate_preference_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args()
    ds = load_dataset("json", data_files=args.data, split="train")
    print(json.dumps(evaluate_preference_accuracy(
        args.model, ds, max_samples=args.max_samples, max_length=args.max_length
    ), indent=2))


if __name__ == "__main__":
    main()
