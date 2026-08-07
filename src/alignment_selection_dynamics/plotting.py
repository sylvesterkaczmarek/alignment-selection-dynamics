from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _series(runs: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([[row[metric] for row in run["history"]] for run in runs], dtype=float)
    x = np.arange(values.shape[1])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    return x, mean, std


def plot_regime_metric(regime_runs: dict[str, list[dict]], metric: str, ylabel: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for regime, runs in regime_runs.items():
        x, mean, std = _series(runs, metric)
        line = ax.plot(x, mean, label=regime.replace("_", " "))[0]
        ax.fill_between(x, mean - std, mean + std, alpha=0.16, color=line.get_color())
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02) if metric in {"verifier_strength", "alignment_score", "capability"} else None
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_shortcut_challenge(runs: list[dict], path: str | Path) -> None:
    x, verifier_mean, verifier_std = _series(runs, "verifier_strength")
    _, bypass_mean, bypass_std = _series(runs, "bypass_strength")
    _, alignment_mean, alignment_std = _series(runs, "alignment_score")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for mean, std, label in [
        (verifier_mean, verifier_std, "verifier strength"),
        (bypass_mean, bypass_std, "bypass strength"),
        (alignment_mean, alignment_std, "alignment score"),
    ]:
        line = ax.plot(x, mean, label=label)[0]
        ax.fill_between(x, mean - std, mean + std, alpha=0.14, color=line.get_color())
    ax.axvline(9.5, linestyle="--", linewidth=1)
    ax.axvline(19.5, linestyle="--", linewidth=1)
    ax.text(14.5, 0.03, "cheap shortcut phase", ha="center", va="bottom")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean population value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
