#!/usr/bin/env python
from __future__ import annotations

import argparse

from posttrainx.training.merge_adapter import merge_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()
    merge_adapter(args.base_model, args.adapter, args.output, args.dtype)


if __name__ == "__main__":
    main()
