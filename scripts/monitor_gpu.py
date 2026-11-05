#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def sample():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.check_output(cmd, text=True).strip().splitlines()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--output", default="outputs/gpu_memory.csv")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "gpu", "memory_used_mib", "memory_total_mib", "utilization_pct"])
        try:
            while True:
                for line in sample():
                    w.writerow([x.strip() for x in line.split(",")])
                f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
