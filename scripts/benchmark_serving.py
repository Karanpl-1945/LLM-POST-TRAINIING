#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json

from posttrainx.config import load_yaml
from posttrainx.serving.benchmark import run_benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/benchmark.yaml")
    args = p.parse_args()
    result = asyncio.run(run_benchmark(load_yaml(args.config)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
