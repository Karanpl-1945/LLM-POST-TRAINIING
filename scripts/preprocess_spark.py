#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrainx.config import load_yaml
from posttrainx.data.spark_dedup import build_spark, minhash_lsh_deduplicate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spark_dedup.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    spark = build_spark(shuffle_partitions=cfg.get("spark", {}).get("shuffle_partitions"))
    inp, out, mh = cfg["input"], cfg["output"], cfg["minhash"]
    reader = spark.read
    if inp["format"] == "json":
        df = reader.json(inp["path"])
    elif inp["format"] == "parquet":
        df = reader.parquet(inp["path"])
    else:
        raise ValueError("input.format must be json or parquet")
    deduped, metrics = minhash_lsh_deduplicate(
        df,
        id_col=inp["id_col"],
        text_col=inp["text_col"],
        ngram=int(mh.get("ngram", 5)),
        num_features=int(mh.get("num_features", 1 << 20)),
        num_hash_tables=int(mh.get("num_hash_tables", 8)),
        jaccard_threshold=float(mh.get("jaccard_similarity_threshold", 0.85)),
    )
    deduped.write.mode("overwrite").format(out.get("format", "parquet")).save(out["path"])
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/dedup_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
