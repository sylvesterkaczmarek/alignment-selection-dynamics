from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from alignment_selection_dynamics.evolution import EvolutionConfig, run_evolution
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import summarize, validate_run_cohort, validate_seeds
from alignment_selection_dynamics.plotting import plot_shortcut_challenge
from alignment_selection_dynamics.provenance import execution_metadata
from experiments._common import argument_parser, experiment_settings


def phase_snapshot(runs: list[dict], generation: int) -> dict[str, dict]:
    validate_run_cohort(runs, expected_regime="shortcut_challenge")
    if type(generation) is not int or not 0 <= generation < len(runs[0]["history"]):
        raise ValueError("Snapshot generation must identify an existing generation")
    keys = ["capability", "alignment_score", "verifier_strength", "bypass_strength", "fitness"]
    return {
        key: summarize(run["history"][generation][key] for run in runs)
        for key in keys
    }


def run(seeds: list[int], out_dir: Path, cfg: EvolutionConfig) -> dict:
    seeds = validate_seeds(seeds)
    if cfg.generations < 30:
        raise ValueError("shortcut_challenge requires at least 30 generations for the fixed three-phase schedule")
    runs = [run_evolution("shortcut_challenge", seed, cfg) for seed in seeds]
    summary = {
        "pre_challenge": phase_snapshot(runs, 9),
        "end_shortcut_phase": phase_snapshot(runs, 19),
        "end_recovery": phase_snapshot(runs, cfg.generations - 1),
    }
    write_json(out_dir / "shortcut_challenge.json", {
        "seeds": seeds, "config": asdict(cfg), "provenance": execution_metadata(), "runs": runs,
    })
    write_json(out_dir / "shortcut_challenge_summary.json", summary)
    plot_shortcut_challenge(runs, out_dir / "figures" / "shortcut_challenge.png")
    return summary


def parse_args() -> argparse.Namespace:
    return argument_parser("Run a cheap-route bypass challenge against verifier selection.").parse_args()


def main() -> None:
    args = parse_args()
    seeds, cfg = experiment_settings(args)
    if cfg.generations < 30:
        raise SystemExit("shortcut_challenge requires at least 30 generations for the fixed three-phase schedule")
    run(seeds, args.out, cfg)


if __name__ == "__main__":
    main()
