from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_runs(a_path: str, b_path: str) -> dict[str, dict[str, Any]]:
    a, b = load_metrics(a_path), load_metrics(b_path)
    keys = sorted(set(a) | set(b))
    result = {}
    for key in keys:
        av, bv = a.get(key), b.get(key)
        delta = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            delta = bv - av
        result[key] = {"run_a": av, "run_b": bv, "b_minus_a": delta}
    return result
