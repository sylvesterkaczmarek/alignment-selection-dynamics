from __future__ import annotations

import argparse
from pathlib import Path

from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json
from experiments.selection_regimes import run as run_selection_regimes
from experiments.shortcut_challenge import run as run_shortcut_challenge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Alignment Selection Dynamics reference suite.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 41, 53])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations < 30:
        raise SystemExit("The full reference suite requires at least 30 generations.")
    cfg = EvolutionConfig(generations=args.generations, population_size=args.population)
    selection = run_selection_regimes(args.seeds, args.out, cfg)
    shortcut = run_shortcut_challenge(args.seeds, args.out, cfg)
    write_json(
        args.out / "reference_summary.json",
        {"seeds": args.seeds, "selection_regimes": selection, "shortcut_challenge": shortcut},
    )


if __name__ == "__main__":
    main()
