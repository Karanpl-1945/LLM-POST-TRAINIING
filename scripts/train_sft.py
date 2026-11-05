#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from posttrainx.config import load_yaml
from posttrainx.training.sft import run_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    print(json.dumps(run_sft(load_yaml(args.config)), indent=2, default=str))


if __name__ == "__main__":
    main()
