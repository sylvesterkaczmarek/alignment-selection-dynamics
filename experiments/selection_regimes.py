from __future__ import annotations

import argparse
from pathlib import Path

from alignment_selection_dynamics.evolution import EvolutionConfig, run_evolution
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import aggregate_final_runs
from alignment_selection_dynamics.plotting import plot_regime_metric

REGIMES = ["safety_only", "neutral", "capability_positive"]


def run(seeds: list[int], out_dir: Path, cfg: EvolutionConfig) -> dict:
    runs_by_regime: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for regime in REGIMES:
        runs = [run_evolution(regime, seed, cfg) for seed in seeds]
        runs_by_regime[regime] = runs
        summary[regime] = aggregate_final_runs(runs)

    write_json(out_dir / "selection_regimes.json", {"seeds": seeds, "runs": runs_by_regime})
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
    parser = argparse.ArgumentParser(description="Compare selection regimes for an alignment-linked verifier trait.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 41, 53])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvolutionConfig(generations=args.generations, population_size=args.population)
    run(args.seeds, args.out, cfg)


if __name__ == "__main__":
    main()
