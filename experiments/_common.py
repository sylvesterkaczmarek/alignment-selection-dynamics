from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.metrics import validate_seeds

DEFAULT_SEEDS = [7, 17, 29, 41, 53]


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="YAML EvolutionConfig fields and seeds")
    parser.add_argument("--seeds", nargs="+", type=int, help="override configuration seeds")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--generations", type=int, help="override configuration generations")
    parser.add_argument("--population", type=int, help="override configuration population size")
    return parser


def experiment_settings(args: argparse.Namespace) -> tuple[list[int], EvolutionConfig]:
    values = {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
            raise ValueError("Configuration must be a YAML mapping with string keys")
        allowed = {field.name for field in fields(EvolutionConfig)} | {"seeds"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    configured_seeds = values.pop("seeds", DEFAULT_SEEDS)
    seeds = args.seeds if args.seeds is not None else configured_seeds
    if not isinstance(seeds, list):
        raise ValueError("Configuration seeds must be a list of distinct integers")
    seeds = validate_seeds(seeds)
    if args.generations is not None:
        values["generations"] = args.generations
    if args.population is not None:
        values["population_size"] = args.population
    return seeds, EvolutionConfig(**values)
