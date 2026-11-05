#!/usr/bin/env python
from __future__ import annotations

import argparse
import math


def main():
    p = argparse.ArgumentParser(description="Back-of-envelope LLM serving capacity estimator")
    p.add_argument("--peak-qps", type=float, required=True)
    p.add_argument("--avg-output-tokens", type=float, required=True)
    p.add_argument("--gpu-output-tokens-per-sec", type=float, required=True,
                   help="Measured per-GPU throughput at the target latency/concurrency")
    p.add_argument("--headroom", type=float, default=1.30)
    p.add_argument("--avg-e2e-seconds", type=float, default=2.0)
    p.add_argument("--avg-cached-tokens", type=int, default=1024)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--kv-bytes", type=float, default=1.0, help="1 for FP8, 2 for BF16/FP16")
    args = p.parse_args()

    required_output_tps = args.peak_qps * args.avg_output_tokens
    theoretical = required_output_tps / args.gpu_output_tokens_per_sec
    provisioned = math.ceil(theoretical * args.headroom)
    concurrency = args.peak_qps * args.avg_e2e_seconds
    kv_bytes = (
        concurrency * args.avg_cached_tokens * args.layers * args.kv_heads * args.head_dim * 2 * args.kv_bytes
    )

    print(f"Required output throughput : {required_output_tps:,.0f} tokens/s")
    print(f"Theoretical GPU replicas   : {theoretical:,.2f}")
    print(f"With headroom              : {provisioned:,} GPU replicas")
    print(f"Estimated concurrency      : {concurrency:,.0f} active requests")
    print(f"Approx aggregate KV bytes  : {kv_bytes / 1024**3:,.2f} GiB")
    print("\nKV estimate is architecture/workload-level arithmetic, not a replacement for engine profiling.")


if __name__ == "__main__":
    main()
