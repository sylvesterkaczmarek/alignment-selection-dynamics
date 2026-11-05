"""Eight paired interventions plus fitness-independent and fixed-trait controls."""

from __future__ import annotations

import hashlib
import gzip
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from alignment_selection_dynamics.causal import (
    PROTOCOL,
    Controls,
    advance,
    checkpoint_fingerprint,
    initial_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from alignment_selection_dynamics.evidence_contracts import validate_manifest, validate_prefix, validate_arm
from alignment_selection_dynamics.environment import EnvironmentSpec
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json, write_bytes
from alignment_selection_dynamics.integrity import canonical, strict_json, finite
from alignment_selection_dynamics.metrics import summarize, validate_seeds
from alignment_selection_dynamics.provenance import execution_metadata
from alignment_selection_dynamics.trajectory import validate_segment
from experiments._common import argument_parser, experiment_settings

BASELINE = EnvironmentSpec(0.25, 0.35, 0.5, 0.5, 0.01, "capability-positive")
METRICS = ("alignment_score", "capability", "verifier_strength", "bypass_strength")


def factorial_conditions() -> dict[str, EnvironmentSpec]:
    conditions = {}
    for bypass, shift, cost in itertools.product((0, 1), repeat=3):
        name = f"b{bypass}_s{shift}_c{cost}"
        conditions[name] = (
            EnvironmentSpec(
                0.0 if shift else 0.25,
                0.0 if shift else 0.35,
                0.995 if bypass else 0.5,
                0.995 if bypass else 0.5,
                0.03 if cost else 0.01,
                "factorial-challenge",
            )
            if (bypass, shift, cost) != (0, 0, 0)
            else BASELINE
        )
    return conditions


def study_arms(split_shift=False):
    if type(split_shift) is not bool:
        raise ValueError("split_shift must be boolean")
    arms = {name: (env, Controls()) for name, env in factorial_conditions().items()}
    for name in ("b0_s0_c0", "b1_s1_c1"):
        arms[f"{name}_random"] = (arms[name][0], Controls(selection="random"))
        arms[f"{name}_fixed_traits"] = (arms[name][0], Controls(mutate_traits=False))
    if split_shift:
        arms["train_shift_only"] = (
            EnvironmentSpec(0.0, 0.35, 0.5, 0.5, 0.01, "train-shift-only"),
            Controls(),
        )
        arms["eval_shift_only"] = (
            EnvironmentSpec(0.25, 0.0, 0.5, 0.5, 0.01, "eval-shift-only"),
            Controls(),
        )
    return arms


def uncertainty(values):
    out = summarize(values)
    out["paired_values"] = list(values)
    out["bootstrap_percentile_95"] = None
    if len(values) > 1:
        rng = np.random.default_rng(260511)
        samples = np.asarray(values)[
            rng.integers(len(values), size=(5000, len(values)))
        ].mean(axis=1)
        out["bootstrap_percentile_95"] = [
            float(v) for v in np.quantile(samples, (0.025, 0.975))
        ]
    return out


def enrich_summary(summary, cohorts, challenge):
    for effect in summary["contrasts"].values():
        for phase in effect.values():
            for metric, estimate in phase.items():
                phase[metric] = uncertainty(estimate["paired_values"])
    if {"train_shift_only", "eval_shift_only"} <= cohorts.keys():
        summaries = {}
        for phase, index in (("end_challenge", challenge - 1), ("end_recovery", -1)):
            summaries[phase] = {}
            for metric in METRICS:
                values = {
                    name: [run["history"][index][metric] for run in cohorts[name]]
                    for name in (
                        "b0_s0_c0",
                        "train_shift_only",
                        "eval_shift_only",
                        "b0_s1_c0",
                    )
                }
                a, b, c, d = (np.asarray(values[name]) for name in values)
                summaries[phase][metric] = {
                    name: uncertainty(effect.tolist())
                    for name, effect in (
                        ("training_shift_removal", ((b - a) + (d - c)) / 2),
                        ("evaluation_shift_removal", ((c - a) + (d - b)) / 2),
                        ("difference_of_differences", d - b - c + a),
                    )
                }
        summary["shift_decomposition"] = summaries
    summary["bootstrap"] = {
        "seed": 260511,
        "resamples": 5000,
        "unit": "independent root seed, paired across conditions",
        "interpretation": "Descriptive percentile intervals with few roots, not multiplicity-adjusted significance tests.",
    }
    return summary


def write_trajectory(path: Path, payload: dict) -> None:
    data = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, gzip.compress(data, mtime=0), exclusive=True)


def read_trajectory(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("compressed trajectory exceeds 64 MiB")
    with gzip.open(path, "rb") as handle:
        raw = handle.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("expanded trajectory exceeds 64 MiB")
    return strict_json(raw)


def summarize_factorial(runs: dict[str, list[dict]], challenge_length: int) -> dict:
    """Root-level paired contrasts; generations and descendants are dependent."""
    result = {"conditions": {}, "contrasts": {}}
    conditions = list(factorial_conditions())
    if any(name not in runs for name in conditions):
        raise ValueError("All eight factorial conditions are required")
    allowed = (
        set(conditions)
        | {
            f"{name}_{suffix}"
            for name in ("b0_s0_c0", "b1_s1_c1")
            for suffix in ("random", "fixed_traits")
        }
        | {"train_shift_only", "eval_shift_only"}
    )
    if runs.keys() - allowed:
        raise ValueError("Unknown experimental arm")
    reference = runs[conditions[0]]
    seeds = validate_seeds(run["seed"] for run in reference)
    if type(challenge_length) is not int or challenge_length < 1:
        raise ValueError("challenge_length must be a positive integer")
    if len(reference[0]["history"]) <= challenge_length:
        raise ValueError("Histories must include challenge and recovery")
    generations = [row["generation"] for row in reference[0]["history"]]
    if any(type(g) is not int or g < 0 for g in generations) or generations != list(
        range(generations[0], generations[0] + len(generations))
    ):
        raise ValueError("History generations must be consecutive")
    for cohort in runs.values():
        if validate_seeds(run["seed"] for run in cohort) != seeds:
            raise ValueError("Conditions must contain the same ordered root seeds")
        if any(
            [row["generation"] for row in run["history"]] != generations
            for run in cohort
        ):
            raise ValueError("Condition histories must cover identical generations")
        for run in cohort:
            for row in run["history"]:
                if type(row["generation"]) is not int:
                    raise ValueError("Generation must be a built-in integer")
                for metric in METRICS:
                    finite(row[metric], metric, probability=True)
    for name, cohort in runs.items():
        result["conditions"][name] = {
            phase: {
                metric: summarize(run["history"][index][metric] for run in cohort)
                for metric in METRICS
            }
            for phase, index in (
                ("end_challenge", challenge_length - 1),
                ("end_recovery", -1),
            )
        }
        deficits = [
            control["history"][-1]["alignment_score"]
            - run["history"][-1]["alignment_score"]
            for control, run in zip(runs["b0_s0_c0"], cohort)
        ]
        result["conditions"][name]["final_probe_deficit_vs_unchanged"] = dict(
            summarize(deficits), paired_values=deficits
        )
    # A k-factor interaction is the k-fold difference, averaged over other factors.
    for size in (1, 2, 3):
        for axes in itertools.combinations(range(3), size):
            name = "_x_".join(("bypass", "shift_removal", "cost")[i] for i in axes)
            result["contrasts"][name] = {}
            for phase, index in (
                ("end_challenge", challenge_length - 1),
                ("end_recovery", -1),
            ):
                result["contrasts"][name][phase] = {}
                for metric in METRICS:
                    paired = []
                    for root in range(len(runs[conditions[0]])):
                        total = 0.0
                        for bits, condition in zip(
                            itertools.product((0, 1), repeat=3), conditions
                        ):
                            sign = np.prod([2 * bits[axis] - 1 for axis in axes])
                            total += (
                                sign * runs[condition][root]["history"][index][metric]
                            )
                        paired.append(float(total / 2 ** (3 - size)))
                    result["contrasts"][name][phase][metric] = dict(
                        summarize(paired), paired_values=paired
                    )
    return result


def run(
    seeds: list[int],
    out: Path,
    cfg: EvolutionConfig,
    *,
    prefix: int = 10,
    challenge: int = 10,
    resume: bool = False,
    split_shift: bool = False,
) -> dict:
    seeds = validate_seeds(seeds)
    arms = study_arms(split_shift)
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if not isinstance(cfg, EvolutionConfig):
        raise TypeError("cfg must be EvolutionConfig")
    if any(type(v) is not int or v < 1 for v in (prefix, challenge)):
        raise ValueError("prefix and challenge durations must be positive integers")
    recovery = cfg.generations - prefix - challenge
    if recovery < 1:
        raise ValueError("The configured horizon must include recovery")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "config": asdict(cfg),
        "prefix": prefix,
        "challenge": challenge,
        "split_shift": split_shift,
    }
    manifest_path = out / "manifest.json"
    if resume:
        manifest = strict_json(manifest_path.read_text())
        validate_manifest(manifest)
        if any(canonical(manifest.get(k)) != canonical(v) for k, v in settings.items()):
            raise ValueError("Resume settings differ from the saved experiment")
        old_hashes = manifest["initial_provenance"]["source_sha256"]
        current = execution_metadata()
        if current["source_sha256"] != old_hashes:
            raise ValueError("Cannot resume after changing simulation source")
        previous = manifest["initial_provenance"]
        if (
            previous["packages"] != current["packages"]
            or previous["python"] != current["python"]
            or previous["platform"] != current["platform"]
            or canonical(previous["torch_settings"]) != canonical(current["torch_settings"])
        ):
            raise ValueError("Cannot resume in a different recorded runtime")
        if (out / "summary.json").exists():
            raise FileExistsError("This experiment is already complete")
        manifest["resume_provenance"].append(current)
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(
            settings, initial_provenance=execution_metadata(), resume_provenance=[]
        )
    write_json(manifest_path, manifest)
    cohorts = {name: [] for name in arms}
    continuation_checks = {}
    for seed in seeds:
        root = out / f"seed-{seed}"
        initial = initial_checkpoint(seed, cfg)
        if resume and (root / "prefix.json.gz").exists():
            shared = load_checkpoint(root / "pre_challenge.pt")
            saved_prefix = read_trajectory(root / "prefix.json.gz")
            fingerprint = checkpoint_fingerprint(shared)
            if (
                saved_prefix["checkpoint_fingerprint"] != fingerprint
                or saved_prefix["config"] != asdict(cfg)
                or shared.next_generation != prefix
            ):
                raise ValueError(
                    "Saved prefix does not match its checkpoint or settings"
                )
            validate_prefix(saved_prefix, seed, cfg, fingerprint)
            pre_history, pre_individuals = (
                saved_prefix["history"],
                saved_prefix["individuals"],
            )
        else:
            shared, pre_history, pre_individuals = advance(
                initial, [BASELINE] * prefix, cfg
            )
            root.mkdir()
            save_checkpoint(root / "pre_challenge.pt", shared)
            shared = load_checkpoint(root / "pre_challenge.pt")
            fingerprint = checkpoint_fingerprint(shared)
            write_trajectory(
                root / "prefix.json.gz",
                {
                    "seed": seed,
                    "config": asdict(cfg),
                    "history": pre_history,
                    "individuals": pre_individuals,
                    "checkpoint_fingerprint": fingerprint,
                },
            )
        validate_segment(
            pre_history,
            pre_individuals,
            initial,
            shared,
            cfg,
            Controls(),
            [BASELINE] * prefix,
        )
        for name, (env, controls) in arms.items():
            target = root / name
            if resume and (target / "trajectory.json.gz").exists():
                payload = read_trajectory(target / "trajectory.json.gz")
                if (
                    payload["seed"] != seed
                    or payload["arm"] != name
                    or payload["config"] != asdict(cfg)
                    or payload["controls"] != asdict(controls)
                    or payload["prefix_checkpoint_fingerprint"] != fingerprint
                ):
                    raise ValueError(
                        "Saved arm does not match the requested comparison"
                    )
                validate_arm(payload, seed, name, cfg, controls, fingerprint)
                end = load_checkpoint(target / "final.pt")
                if end.next_generation != cfg.generations:
                    raise ValueError("Saved arm has an incomplete final checkpoint")
                middle = load_checkpoint(target / "pre_recovery.pt")
                if payload.get(
                    "final_checkpoint_fingerprint"
                ) != checkpoint_fingerprint(end) or payload.get(
                    "challenge_checkpoint_fingerprint"
                ) != checkpoint_fingerprint(middle):
                    raise ValueError("saved arm checkpoint fingerprint mismatch")
                validate_segment(
                    payload["history"][:challenge],
                    payload["individuals"][: challenge * cfg.population_size],
                    shared,
                    middle,
                    cfg,
                    controls,
                    [env] * challenge,
                )
                validate_segment(
                    payload["history"][challenge:],
                    payload["individuals"][challenge * cfg.population_size :],
                    middle,
                    end,
                    cfg,
                    controls,
                    [BASELINE] * recovery,
                )
                cohorts[name].append({"seed": seed, "history": payload["history"]})
                if name == "b0_s0_c0":
                    uninterrupted, whole, whole_records = advance(
                        initial, [BASELINE] * cfg.generations, cfg
                    )
                    if (
                        whole != pre_history + payload["history"]
                        or whole_records != pre_individuals + payload["individuals"]
                        or checkpoint_fingerprint(uninterrupted)
                        != checkpoint_fingerprint(end)
                    ):
                        raise AssertionError("Saved unchanged continuation diverged")
                    continuation_checks[str(seed)] = True
                continue
            end_challenge, first, records_a = advance(
                shared, [env] * challenge, cfg, controls
            )
            end, second, records_b = advance(
                end_challenge, [BASELINE] * recovery, cfg, controls
            )
            target = root / name
            save_checkpoint(target / "pre_recovery.pt", end_challenge)
            save_checkpoint(target / "final.pt", end)
            validate_segment(
                first,
                records_a,
                shared,
                end_challenge,
                cfg,
                controls,
                [env] * challenge,
            )
            validate_segment(
                second,
                records_b,
                end_challenge,
                end,
                cfg,
                controls,
                [BASELINE] * recovery,
            )
            payload = {
                "final_checkpoint_fingerprint": checkpoint_fingerprint(end),
                "challenge_checkpoint_fingerprint": checkpoint_fingerprint(
                    end_challenge
                ),
                "protocol": PROTOCOL,
                "seed": seed,
                "arm": name,
                "config": asdict(cfg),
                "controls": asdict(controls),
                "prefix_checkpoint_fingerprint": fingerprint,
                "history": first + second,
                "individuals": records_a + records_b,
            }
            write_trajectory(target / "trajectory.json.gz", payload)
            cohorts[name].append({"seed": seed, "history": payload["history"]})
            if name == "b0_s0_c0":
                uninterrupted, whole, whole_records = advance(
                    initial, [BASELINE] * cfg.generations, cfg
                )
                if (
                    whole != pre_history + first + second
                    or whole_records != pre_individuals + records_a + records_b
                    or checkpoint_fingerprint(uninterrupted)
                    != checkpoint_fingerprint(end)
                ):
                    raise AssertionError("Unchanged checkpoint continuation diverged")
                continuation_checks[str(seed)] = True
        print(
            f"seed {seed}: {len(arms)} arms completed; continuation identical",
            flush=True,
        )
    summary = enrich_summary(
        summarize_factorial(cohorts, challenge), cohorts, challenge
    )
    summary.update(
        {
            "protocol": PROTOCOL,
            "study_type": "exploratory",
            "seeds": seeds,
            "config": asdict(cfg),
            "split_shift": split_shift,
            "prefix_generations": prefix,
            "challenge_generations": challenge,
            "recovery_generations": recovery,
            "continuation_checks": continuation_checks,
            "arms": {
                name: {"environment": asdict(env), "controls": asdict(controls)}
                for name, (env, controls) in arms.items()
            },
            "provenance": execution_metadata(),
            "uncertainty": "std is sample standard deviation across independent root seeds, not a confidence interval",
            "contrast_definition": "k-fold high-minus-low difference, averaged over remaining factors",
        }
    )
    write_json(out / "summary.json", summary)
    files = {
        path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "completion.json"
    }
    write_json(
        out / "completion.json",
        {
            "schema_version": 1,
            "status": "complete",
            "files": files,
            "roots": seeds,
            "arms": list(arms),
        },
        exclusive=True,
    )
    return summary


def main() -> None:
    parser = argument_parser("Run the paired causal factorial and controls.")
    parser.set_defaults(out=Path("results/local/causal-v1"))
    parser.add_argument(
        "--split-shift",
        action="store_true",
        help="also isolate training and evaluation shift removal",
    )
    parser.add_argument("--prefix", type=int, default=10)
    parser.add_argument("--challenge", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an incomplete run with unchanged simulation source",
    )
    args = parser.parse_args()
    seeds, cfg = experiment_settings(args)
    run(
        seeds,
        args.out,
        cfg,
        prefix=args.prefix,
        challenge=args.challenge,
        resume=args.resume,
        split_shift=args.split_shift,
    )


if __name__ == "__main__":
    main()
