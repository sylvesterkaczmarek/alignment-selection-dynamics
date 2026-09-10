"""Eight paired interventions plus fitness-independent and fixed-trait controls."""
from __future__ import annotations

import gzip
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from alignment_selection_dynamics.causal import (
    PROTOCOL, Controls, advance, checkpoint_fingerprint, initial_checkpoint,
    load_checkpoint, save_checkpoint,
)
from alignment_selection_dynamics.environment import EnvironmentSpec
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import summarize, validate_seeds
from alignment_selection_dynamics.provenance import execution_metadata
from experiments._common import argument_parser, experiment_settings

BASELINE = EnvironmentSpec(0.25, 0.35, 0.5, 0.5, 0.01, "capability-positive")
METRICS = ("alignment_score", "capability", "verifier_strength", "bypass_strength")


def factorial_conditions() -> dict[str, EnvironmentSpec]:
    conditions = {}
    for bypass, shift, cost in itertools.product((0, 1), repeat=3):
        name = f"b{bypass}_s{shift}_c{cost}"
        conditions[name] = EnvironmentSpec(
            0.0 if shift else 0.25, 0.0 if shift else 0.35,
            0.995 if bypass else 0.5, 0.995 if bypass else 0.5,
            0.03 if cost else 0.01, "factorial-challenge",
        ) if (bypass, shift, cost) != (0, 0, 0) else BASELINE
    return conditions


def write_trajectory(path: Path, payload: dict) -> None:
    data = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(gzip.compress(data, mtime=0))


def read_trajectory(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def summarize_factorial(runs: dict[str, list[dict]], challenge_length: int) -> dict:
    """Root-level paired contrasts; generations and descendants are dependent."""
    result = {"conditions": {}, "contrasts": {}}
    conditions = list(factorial_conditions())
    if any(name not in runs for name in conditions):
        raise ValueError("All eight factorial conditions are required")
    reference = runs[conditions[0]]
    seeds = validate_seeds(run["seed"] for run in reference)
    if type(challenge_length) is not int or challenge_length < 1:
        raise ValueError("challenge_length must be a positive integer")
    if len(reference[0]["history"]) <= challenge_length:
        raise ValueError("Histories must include challenge and recovery")
    generations = [row["generation"] for row in reference[0]["history"]]
    if (any(type(g) is not int or g < 0 for g in generations) or
            generations != list(range(generations[0], generations[0] + len(generations)))):
        raise ValueError("History generations must be consecutive")
    for cohort in runs.values():
        if validate_seeds(run["seed"] for run in cohort) != seeds:
            raise ValueError("Conditions must contain the same ordered root seeds")
        if any([row["generation"] for row in run["history"]] != generations for run in cohort):
            raise ValueError("Condition histories must cover identical generations")
    for name, cohort in runs.items():
        result["conditions"][name] = {
            phase: {metric: summarize(run["history"][index][metric] for run in cohort)
                    for metric in METRICS}
            for phase, index in (("end_challenge", challenge_length - 1), ("end_recovery", -1))
        }
        deficits = [control["history"][-1]["alignment_score"] - run["history"][-1]["alignment_score"]
                    for control, run in zip(runs["b0_s0_c0"], cohort)]
        result["conditions"][name]["final_probe_deficit_vs_unchanged"] = dict(
            summarize(deficits), paired_values=deficits)
    # A k-factor interaction is the k-fold difference, averaged over other factors.
    for size in (1, 2, 3):
        for axes in itertools.combinations(range(3), size):
            name = "_x_".join(("bypass", "shift_removal", "cost")[i] for i in axes)
            result["contrasts"][name] = {}
            for phase, index in (("end_challenge", challenge_length - 1), ("end_recovery", -1)):
                result["contrasts"][name][phase] = {}
                for metric in METRICS:
                    paired = []
                    for root in range(len(runs[conditions[0]])):
                        total = 0.0
                        for bits, condition in zip(itertools.product((0, 1), repeat=3), conditions):
                            sign = np.prod([2 * bits[axis] - 1 for axis in axes])
                            total += sign * runs[condition][root]["history"][index][metric]
                        paired.append(float(total / 2 ** (3 - size)))
                    result["contrasts"][name][phase][metric] = dict(
                        summarize(paired), paired_values=paired)
    return result


def run(seeds: list[int], out: Path, cfg: EvolutionConfig, *, prefix: int = 10,
        challenge: int = 10, resume: bool = False) -> dict:
    seeds = validate_seeds(seeds)
    if any(type(v) is not int or v < 1 for v in (prefix, challenge)):
        raise ValueError("prefix and challenge durations must be positive integers")
    recovery = cfg.generations - prefix - challenge
    if recovery < 1:
        raise ValueError("The configured horizon must include recovery")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = {"protocol": PROTOCOL, "seeds": seeds, "config": asdict(cfg),
                "prefix": prefix, "challenge": challenge}
    manifest_path = out / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[k] != v for k, v in settings.items()):
            raise ValueError("Resume settings differ from the saved experiment")
        old_hashes = manifest["initial_provenance"]["source_sha256"]
        current = execution_metadata()
        for name, digest in old_hashes.items():
            if (name.startswith("alignment_selection_dynamics/") or name == "experiments/causal_factorial.py") and current["source_sha256"].get(name) != digest:
                raise ValueError("Cannot resume after changing simulation source")
        previous = manifest["initial_provenance"]
        if (previous["packages"] != current["packages"] or
                previous["python"] != current["python"] or previous["platform"] != current["platform"]):
            raise ValueError("Cannot resume in a different recorded runtime")
        if (out / "summary.json").exists():
            raise FileExistsError("This experiment is already complete")
        manifest["resume_provenance"].append(current)
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(settings, initial_provenance=execution_metadata(), resume_provenance=[])
    write_json(manifest_path, manifest)
    arms = {name: (env, Controls()) for name, env in factorial_conditions().items()}
    for name in ("b0_s0_c0", "b1_s1_c1"):
        arms[f"{name}_random"] = (arms[name][0], Controls(selection="random"))
        arms[f"{name}_fixed_traits"] = (arms[name][0], Controls(mutate_traits=False))
    cohorts = {name: [] for name in arms}
    continuation_checks = {}
    for seed in seeds:
        root = out / f"seed-{seed}"
        initial = initial_checkpoint(seed, cfg)
        if resume and (root / "prefix.json.gz").exists():
            shared = load_checkpoint(root / "pre_challenge.pt")
            saved_prefix = read_trajectory(root / "prefix.json.gz")
            fingerprint = checkpoint_fingerprint(shared)
            if (saved_prefix["checkpoint_fingerprint"] != fingerprint or
                    saved_prefix["config"] != asdict(cfg) or shared.next_generation != prefix):
                raise ValueError("Saved prefix does not match its checkpoint or settings")
            pre_history, pre_individuals = saved_prefix["history"], saved_prefix["individuals"]
        else:
            shared, pre_history, pre_individuals = advance(initial, [BASELINE] * prefix, cfg)
            root.mkdir()
            save_checkpoint(root / "pre_challenge.pt", shared)
            shared = load_checkpoint(root / "pre_challenge.pt")
            fingerprint = checkpoint_fingerprint(shared)
            write_trajectory(root / "prefix.json.gz", {
                "seed": seed, "config": asdict(cfg), "history": pre_history,
                "individuals": pre_individuals, "checkpoint_fingerprint": fingerprint,
            })
        for name, (env, controls) in arms.items():
            target = root / name
            if resume and (target / "trajectory.json.gz").exists():
                payload = read_trajectory(target / "trajectory.json.gz")
                if (payload["seed"] != seed or payload["arm"] != name or
                        payload["config"] != asdict(cfg) or payload["controls"] != asdict(controls) or
                        payload["prefix_checkpoint_fingerprint"] != fingerprint):
                    raise ValueError("Saved arm does not match the requested comparison")
                end = load_checkpoint(target / "final.pt")
                if end.next_generation != cfg.generations:
                    raise ValueError("Saved arm has an incomplete final checkpoint")
                cohorts[name].append({"seed": seed, "history": payload["history"]})
                if name == "b0_s0_c0":
                    uninterrupted, whole, whole_records = advance(initial, [BASELINE] * cfg.generations, cfg)
                    if (whole != pre_history + payload["history"] or
                            whole_records != pre_individuals + payload["individuals"] or
                            checkpoint_fingerprint(uninterrupted) != checkpoint_fingerprint(end)):
                        raise AssertionError("Saved unchanged continuation diverged")
                    continuation_checks[str(seed)] = True
                continue
            end_challenge, first, records_a = advance(shared, [env] * challenge, cfg, controls)
            end, second, records_b = advance(end_challenge, [BASELINE] * recovery, cfg, controls)
            target = root / name
            save_checkpoint(target / "pre_recovery.pt", end_challenge)
            save_checkpoint(target / "final.pt", end)
            payload = {
                "protocol": PROTOCOL, "seed": seed, "arm": name, "config": asdict(cfg),
                "controls": asdict(controls), "prefix_checkpoint_fingerprint": fingerprint,
                "history": first + second, "individuals": records_a + records_b,
            }
            write_trajectory(target / "trajectory.json.gz", payload)
            cohorts[name].append({"seed": seed, "history": payload["history"]})
            if name == "b0_s0_c0":
                uninterrupted, whole, whole_records = advance(initial, [BASELINE] * cfg.generations, cfg)
                if (whole != pre_history + first + second or
                        whole_records != pre_individuals + records_a + records_b or
                        checkpoint_fingerprint(uninterrupted) != checkpoint_fingerprint(end)):
                    raise AssertionError("Unchanged checkpoint continuation diverged")
                continuation_checks[str(seed)] = True
        print(f"seed {seed}: {len(arms)} arms completed; continuation identical", flush=True)
    summary = summarize_factorial(cohorts, challenge)
    summary.update({
        "protocol": PROTOCOL, "study_type": "exploratory", "seeds": seeds,
        "config": asdict(cfg), "prefix_generations": prefix, "challenge_generations": challenge,
        "recovery_generations": recovery, "continuation_checks": continuation_checks,
        "arms": {name: {"environment": asdict(env), "controls": asdict(controls)}
                 for name, (env, controls) in arms.items()},
        "provenance": execution_metadata(),
        "uncertainty": "std is sample standard deviation across independent root seeds, not a confidence interval",
        "contrast_definition": "k-fold high-minus-low difference, averaged over remaining factors",
    })
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argument_parser("Run the paired causal factorial and controls.")
    parser.set_defaults(out=Path("results/local/causal-v1"))
    parser.add_argument("--prefix", type=int, default=10)
    parser.add_argument("--challenge", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="continue an incomplete run with unchanged simulation source")
    args = parser.parse_args()
    seeds, cfg = experiment_settings(args)
    run(seeds, args.out, cfg, prefix=args.prefix, challenge=args.challenge, resume=args.resume)


if __name__ == "__main__":
    main()
