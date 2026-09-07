from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from alignment_selection_dynamics.evolution import EvolutionConfig, run_evolution
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import aggregate_final_runs, validate_seeds
from alignment_selection_dynamics.plotting import plot_regime_metric
from alignment_selection_dynamics.provenance import execution_metadata
from experiments._common import argument_parser, experiment_settings

REGIMES = ["safety_only", "neutral", "capability_positive"]


def run(seeds: list[int], out_dir: Path, cfg: EvolutionConfig) -> dict:
    seeds = validate_seeds(seeds)
    runs_by_regime: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for regime in REGIMES:
        runs = [run_evolution(regime, seed, cfg) for seed in seeds]
        runs_by_regime[regime] = runs
        summary[regime] = aggregate_final_runs(runs)

    write_json(out_dir / "selection_regimes.json", {
        "seeds": seeds, "config": asdict(cfg), "provenance": execution_metadata(), "runs": runs_by_regime,
    })
    write_json(out_dir / "selection_regimes_summary.json", summary)
    plot_regime_metric(
        runs_by_regime,
        "verifier_strength",
        "Mean verifier strength",
        out_dir / "figures" / "verifier_selection_dynamics.png",
    )
    plot_regime_metric(
        runs_by_regime,
        "alignment_score",
        "Alignment probe accuracy",
        out_dir / "figures" / "alignment_selection_dynamics.png",
    )
    plot_regime_metric(
        runs_by_regime,
        "capability",
        "Capability accuracy",
        out_dir / "figures" / "capability_dynamics.png",
    )
    return summary


def parse_args() -> argparse.Namespace:
    return argument_parser("Compare selection regimes for an alignment-linked verifier trait.").parse_args()


def main() -> None:
    args = parse_args()
    seeds, cfg = experiment_settings(args)
    run(seeds, args.out, cfg)


if __name__ == "__main__":
    main()
