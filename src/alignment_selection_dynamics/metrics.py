from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def aggregate_final_runs(runs: list[dict]) -> dict[str, dict]:
    metrics = [
        "fitness",
        "capability",
        "alignment_score",
        "verifier_strength",
        "bypass_strength",
        "selection_differential",
    ]
    out: dict[str, dict] = {}
    for metric in metrics:
        out[metric] = summarize(run["history"][-1][metric] for run in runs)
    return out
