from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RequestMetric:
    ok: bool
    ttft_s: float | None
    e2e_s: float
    output_tokens: int
    error: str | None = None


async def _one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> RequestMetric:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.perf_counter()
    first_token_at = None
    output_tokens = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") and first_token_at is None:
                        first_token_at = time.perf_counter()
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
        end = time.perf_counter()
        return RequestMetric(True, None if first_token_at is None else first_token_at - start, end - start, output_tokens)
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:  # record request and malformed-response failures
        end = time.perf_counter()
        return RequestMetric(False, None, end - start, 0, repr(exc))


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(round((len(xs) - 1) * p), len(xs) - 1)
    return xs[idx]


async def run_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    server = cfg["server"]
    bench = cfg["benchmark"]
    prompts = cfg["prompts"]
    concurrency = int(bench.get("concurrency", 8))
    repeats = int(bench.get("repeats", 1))
    timeout = httpx.Timeout(float(bench.get("timeout_s", 120.0)))
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def wrapped(prompt: str):
            async with sem:
                return await _one_request(
                    client,
                    url=server["base_url"].rstrip("/") + "/chat/completions",
                    api_key=server.get("api_key", "local-dev-key"),
                    model=server["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(bench.get("max_tokens", 128)),
                    temperature=float(bench.get("temperature", 0.0)),
                )

        jobs = [wrapped(p) for _ in range(repeats) for p in prompts]
        wall_start = time.perf_counter()
        metrics = await asyncio.gather(*jobs)
        wall = time.perf_counter() - wall_start

    good = [m for m in metrics if m.ok]
    ttfts = [m.ttft_s for m in good if m.ttft_s is not None]
    e2es = [m.e2e_s for m in good]
    total_tokens = sum(m.output_tokens for m in good)
    summary = {
        "requests": len(metrics),
        "successful_requests": len(good),
        "failed_requests": len(metrics) - len(good),
        "wall_time_s": wall,
        "requests_per_second": len(good) / max(wall, 1e-9),
        "output_tokens_per_second": total_tokens / max(wall, 1e-9),
        "total_output_tokens": total_tokens,
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "e2e_mean_s": statistics.mean(e2es) if e2es else None,
        "e2e_p50_s": _percentile(e2es, 0.50),
        "e2e_p95_s": _percentile(e2es, 0.95),
        "raw": [asdict(m) for m in metrics],
    }
    output = Path(bench.get("output", "outputs/serving_benchmark.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
