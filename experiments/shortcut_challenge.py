from __future__ import annotations

import argparse
from pathlib import Path

from alignment_selection_dynamics.evolution import EvolutionConfig, run_evolution
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import summarize
from alignment_selection_dynamics.plotting import plot_shortcut_challenge


def phase_snapshot(runs: list[dict], generation: int) -> dict[str, dict]:
    keys = ["capability", "alignment_score", "verifier_strength", "bypass_strength", "fitness"]
    return {
        key: summarize(run["history"][generation][key] for run in runs)
        for key in keys
    }


def run(seeds: list[int], out_dir: Path, cfg: EvolutionConfig) -> dict:
    runs = [run_evolution("shortcut_challenge", seed, cfg) for seed in seeds]
    summary = {
        "pre_challenge": phase_snapshot(runs, 9),
        "end_shortcut_phase": phase_snapshot(runs, 19),
        "end_recovery": phase_snapshot(runs, cfg.generations - 1),
    }
    write_json(out_dir / "shortcut_challenge.json", {"seeds": seeds, "runs": runs})
    write_json(out_dir / "shortcut_challenge_summary.json", summary)
    plot_shortcut_challenge(runs, out_dir / "figures" / "shortcut_challenge.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cheap-route bypass challenge against verifier selection.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 41, 53])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations < 30:
        raise SystemExit("shortcut_challenge requires at least 30 generations for the fixed three-phase schedule")
    cfg = EvolutionConfig(generations=args.generations, population_size=args.population)
    run(args.seeds, args.out, cfg)


if __name__ == "__main__":
    main()
