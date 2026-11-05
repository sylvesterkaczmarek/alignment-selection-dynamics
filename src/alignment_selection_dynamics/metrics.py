from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .seeding import validate_seed
from .integrity import canonical, finite
from .evolution import EvolutionConfig
from dataclasses import asdict


def validate_seeds(seeds: Iterable[int]) -> list[int]:
    """Require independent, valid root seeds before an experiment starts."""
    values = list(seeds)
    if not values:
        raise ValueError("At least one seed is required")
    for seed in values:
        validate_seed(seed)
    if len(set(values)) != len(values):
        raise ValueError(
            "Seeds must be distinct; repeated seeds are not independent runs"
        )
    return values


def validate_run_cohort(
    runs: list[dict], *, expected_regime: str | None = None
) -> None:
    """Reject comparisons between different experiments or incomplete histories."""
    if not runs:
        raise ValueError("At least one run is required")
    validate_seeds(run["seed"] for run in runs)
    regime = runs[0]["regime"]
    config = runs[0]["config"]
    if regime not in (
        "safety_only",
        "neutral",
        "capability_positive",
        "shortcut_challenge",
    ):
        raise ValueError("unknown experiment regime")
    if (
        not isinstance(config, dict)
        or config.keys() != asdict(EvolutionConfig()).keys()
    ):
        raise ValueError("incomplete experiment configuration")
    EvolutionConfig(**config)
    if expected_regime is not None and regime != expected_regime:
        raise ValueError(f"Expected {expected_regime} runs")
    for run in runs:
        EvolutionConfig(**run["config"])
        if run["regime"] != regime or canonical(run["config"]) != canonical(config):
            raise ValueError("Runs must use the same regime and configuration")
        history = run["history"]
        if not history or len(history) != config["generations"]:
            raise ValueError("Each run must contain its complete configured history")
        if any(
            type(row["generation"]) is not int or row["generation"] != index
            for index, row in enumerate(history)
        ):
            raise ValueError(
                "History generations must be consecutive and start at zero"
            )
        for row in history:
            for key in (
                "capability",
                "alignment_score",
                "verifier_strength",
                "bypass_strength",
            ):
                finite(row[key], key, probability=True)
            for key in ("fitness", "selection_differential"):
                finite(row[key], key)
            if "verification_cost" in row:
                finite(row["verification_cost"], "verification_cost")
                if (
                    row["verification_cost"] < 0
                    or abs(
                        row["fitness"] - (row["capability"] - row["verification_cost"])
                    )
                    > 1e-12
                ):
                    raise ValueError("fitness/cost evidence is inconsistent")


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    raw = list(values)
    for value in raw:
        finite(value, "summary value")
    array = np.asarray(raw, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Summary values must be finite scalars")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(array.mean())
        std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if not np.isfinite([mean, std]).all():
        raise ValueError("Summary statistics exceed the supported numeric range")
    return {
        "mean": mean,
        "std": std,
        "n": int(array.size),
    }


def aggregate_final_runs(runs: list[dict]) -> dict[str, dict]:
    validate_run_cohort(runs)
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
