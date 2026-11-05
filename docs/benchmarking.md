# Benchmark methodology

Keep the model, prompt set, decoding parameters, hardware, and concurrency fixed when comparing two serving configurations.

Measure at minimum:

- output tokens / second
- requests / second
- TTFT p50 / p95
- end-to-end latency p50 / p95
- peak GPU memory
- error rate
- concurrency

Run a warm-up before recording numbers. Repeat the benchmark and report the workload shape (prompt-length and output-length distributions). Use `scripts/monitor_gpu.py` alongside `scripts/benchmark_serving.py` to capture GPU memory and utilization.

Calculate the throughput multiplier:

```text
throughput_multiplier = optimized_output_tokens_per_second / baseline_output_tokens_per_second
```

Calculate the peak GPU memory reduction:

```text
memory_reduction_pct = (baseline_peak - optimized_peak) / baseline_peak * 100
```

Do not attribute the full improvement to one optimization unless you ran an ablation study. A useful ablation sequence is:

1. baseline server
2. + vLLM scheduling / continuous batching
3. + prefix caching
4. + FP8 KV cache
5. + model-weight quantization
