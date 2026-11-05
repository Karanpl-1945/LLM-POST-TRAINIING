#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from posttrainx.evaluation.training_compare import compare_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    args = p.parse_args()
    print(json.dumps(compare_runs(args.run_a, args.run_b), indent=2))


if __name__ == "__main__":
    main()
