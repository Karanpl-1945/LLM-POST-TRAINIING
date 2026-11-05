from __future__ import annotations

import time
from typing import Any

from pyspark.ml.feature import HashingTF, MinHashLSH, NGram, RegexTokenizer
from pyspark.sql import DataFrame, SparkSession, functions as F


def build_spark(app_name: str = "posttrainx-dedup", shuffle_partitions: int | None = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if shuffle_partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    return builder.getOrCreate()


def normalize_text(df: DataFrame, text_col: str) -> DataFrame:
    return (
        df.withColumn(text_col, F.lower(F.col(text_col)))
        .withColumn(text_col, F.regexp_replace(F.col(text_col), r"\\s+", " "))
        .withColumn(text_col, F.trim(F.col(text_col)))
        .filter(F.length(F.col(text_col)) > 0)
    )


def minhash_lsh_deduplicate(
    df: DataFrame,
    *,
    id_col: str,
    text_col: str,
    ngram: int = 5,
    num_features: int = 1 << 20,
    num_hash_tables: int = 8,
    jaccard_threshold: float = 0.85,
) -> tuple[DataFrame, dict[str, Any]]:
    """Approximate near-duplicate removal using token n-grams + MinHash LSH.

    Notes:
    - MinHashLSH returns Jaccard *distance*, so distance <= (1 - similarity_threshold).
    - This implementation removes the higher-ID member of each detected pair. For datasets
      with complicated transitive duplicate clusters, use a connected-components pass if you
      need one canonical representative per full cluster.
    """
    start = time.perf_counter()
    df = normalize_text(df, text_col).dropDuplicates([id_col])
    before = df.count()

    tokenizer = RegexTokenizer(
        inputCol=text_col,
        outputCol="_tokens",
        pattern=r"\\W+",
        gaps=True,
        minTokenLength=1,
    )
    ngrammer = NGram(n=ngram, inputCol="_tokens", outputCol="_shingles")
    hashing = HashingTF(
        inputCol="_shingles",
        outputCol="features",
        numFeatures=num_features,
        binary=True,
    )

    featurized = tokenizer.transform(df)
    featurized = ngrammer.transform(featurized)
    featurized = hashing.transform(featurized).filter(F.size(F.col("_shingles")) > 0).cache()

    lsh = MinHashLSH(
        inputCol="features",
        outputCol="hashes",
        numHashTables=num_hash_tables,
    ).fit(featurized)

    distance_threshold = 1.0 - jaccard_threshold
    pairs = (
        lsh.approxSimilarityJoin(featurized.alias("a"), featurized.alias("b"), distance_threshold, distCol="jaccard_distance")
        .filter(F.col(f"datasetA.{id_col}") < F.col(f"datasetB.{id_col}"))
        .select(F.col(f"datasetB.{id_col}").alias(id_col))
        .distinct()
    )

    deduped = (
        featurized.join(pairs.withColumn("_is_duplicate", F.lit(1)), on=id_col, how="left")
        .filter(F.col("_is_duplicate").isNull())
        .drop("_is_duplicate", "_tokens", "_shingles", "features", "hashes")
    )

    after = deduped.count()
    elapsed = max(time.perf_counter() - start, 1e-9)
    metrics = {
        "records_before": before,
        "records_after": after,
        "records_removed": before - after,
        "elapsed_seconds": elapsed,
        "records_per_second": before / elapsed,
        "jaccard_similarity_threshold": jaccard_threshold,
    }
    return deduped, metrics
