from __future__ import annotations

import argparse
from dataclasses import asdict

from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.provenance import execution_metadata
from experiments._common import argument_parser, experiment_settings
from experiments.selection_regimes import run as run_selection_regimes
from experiments.shortcut_challenge import run as run_shortcut_challenge


def parse_args() -> argparse.Namespace:
    return argument_parser("Run the full Alignment Selection Dynamics reference suite.").parse_args()


def main() -> None:
    args = parse_args()
    seeds, cfg = experiment_settings(args)
    if cfg.generations < 30:
        raise SystemExit("The full reference suite requires at least 30 generations.")
    selection = run_selection_regimes(seeds, args.out, cfg)
    shortcut = run_shortcut_challenge(seeds, args.out, cfg)
    write_json(
        args.out / "reference_summary.json",
        {"seeds": seeds, "config": asdict(cfg), "provenance": execution_metadata(),
         "selection_regimes": selection, "shortcut_challenge": shortcut},
    )


if __name__ == "__main__":
    main()
