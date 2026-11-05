"""Verify complete factorial evidence and recompute seed-level summaries."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from alignment_selection_dynamics.causal import (
    Controls,
    initial_checkpoint,
    load_checkpoint,
    checkpoint_fingerprint,
    PROTOCOL,
)
from alignment_selection_dynamics.evidence_contracts import validate_manifest, validate_prefix, validate_arm
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.integrity import canonical, strict_json
from alignment_selection_dynamics.metrics import validate_seeds
from alignment_selection_dynamics.provenance import execution_metadata
from alignment_selection_dynamics.trajectory import validate_segment
from .causal_factorial import (
    BASELINE,
    study_arms,
    read_trajectory,
    summarize_factorial,
    enrich_summary,
)


def read_json(path):
    if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("evidence must be a bounded ordinary local file")
    return strict_json(path.read_bytes())


def verify(directory):
    root = Path(directory).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    fields = {
        "protocol",
        "seeds",
        "config",
        "prefix",
        "challenge",
        "split_shift",
        "initial_provenance",
        "resume_provenance",
    }
    if (
        type(manifest) is not dict
        or manifest.keys() != fields
        or manifest["protocol"] != PROTOCOL
    ):
        raise ValueError("unsupported study manifest")
    validate_manifest(manifest)
    seeds = validate_seeds(manifest["seeds"])
    cfg = EvolutionConfig(**manifest["config"])
    if manifest["config"].keys() != asdict(cfg).keys():
        raise ValueError("incomplete study config")
    prefix, challenge = manifest["prefix"], manifest["challenge"]
    recovery = cfg.generations - prefix - challenge
    if any(type(v) is not int or v < 1 for v in (prefix, challenge, recovery)):
        raise ValueError("invalid study phase durations")
    arms = study_arms(manifest["split_shift"])
    expected = {"manifest.json", "summary.json"}
    for seed in seeds:
        folder = f"seed-{seed}"
        expected |= {f"{folder}/pre_challenge.pt", f"{folder}/prefix.json.gz"}
        expected |= {
            f"{folder}/{arm}/{name}"
            for arm in arms
            for name in ("pre_recovery.pt", "final.pt", "trajectory.json.gz")
        }
    completion = read_json(root / "completion.json")
    if (
        completion.keys() != {"schema_version", "status", "files", "roots", "arms"}
        or type(completion["schema_version"]) is not int
        or completion["schema_version"] != 1
        or completion["status"] != "complete"
        or canonical(completion["roots"]) != canonical(seeds)
        or completion["arms"] != list(arms)
        or set(completion["files"]) != expected
    ):
        raise ValueError("incomplete study file manifest")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected | {"completion.json"}:
        raise ValueError("unexpected or missing study files")
    for name in expected:
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            raise ValueError("unsafe study artifact path")
        if completion["files"][name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("artifact checksum mismatch")
    cohorts = {name: [] for name in arms}
    records = 0
    for seed in seeds:
        folder = root / f"seed-{seed}"
        initial = initial_checkpoint(seed, cfg)
        shared = load_checkpoint(folder / "pre_challenge.pt")
        pre = read_trajectory(folder / "prefix.json.gz")
        if (
            pre["seed"] != seed
            or canonical(pre["config"]) != canonical(asdict(cfg))
            or pre["checkpoint_fingerprint"] != checkpoint_fingerprint(shared)
            or shared.next_generation != prefix
        ):
            raise ValueError("prefix identity mismatch")
        validate_prefix(pre, seed, cfg, checkpoint_fingerprint(shared))
        validate_segment(
            pre["history"],
            pre["individuals"],
            initial,
            shared,
            cfg,
            Controls(),
            [BASELINE] * prefix,
        )
        records += len(pre["individuals"])
        for arm, (env, controls) in arms.items():
            path = folder / arm
            payload = read_trajectory(path / "trajectory.json.gz")
            middle = load_checkpoint(path / "pre_recovery.pt")
            end = load_checkpoint(path / "final.pt")
            if (
                payload["protocol"] != PROTOCOL
                or type(payload["seed"]) is not int
                or payload["seed"] != seed
                or payload["arm"] != arm
                or canonical(payload["config"]) != canonical(asdict(cfg))
                or canonical(payload["controls"]) != canonical(asdict(controls))
                or payload["prefix_checkpoint_fingerprint"]
                != checkpoint_fingerprint(shared)
                or payload["challenge_checkpoint_fingerprint"]
                != checkpoint_fingerprint(middle)
                or payload["final_checkpoint_fingerprint"]
                != checkpoint_fingerprint(end)
                or middle.next_generation != prefix + challenge
                or end.next_generation != cfg.generations
            ):
                raise ValueError("arm identity or checkpoint binding mismatch")
            validate_arm(payload, seed, arm, cfg, controls, checkpoint_fingerprint(shared))
            for state in (shared, middle, end):
                if (
                    canonical(state.config) != canonical(asdict(cfg))
                    or state.seed != seed
                ):
                    raise ValueError("checkpoint study mismatch")
            for state in (middle, end):
                if canonical(state.controls) != canonical(asdict(controls)):
                    raise ValueError("checkpoint control mismatch")
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
            cohorts[arm].append({"seed": seed, "history": payload["history"]})
            records += len(payload["individuals"])
    computed = enrich_summary(
        summarize_factorial(cohorts, challenge), cohorts, challenge
    )
    computed.update(
        {
            "protocol": PROTOCOL,
            "study_type": "exploratory",
            "seeds": seeds,
            "config": asdict(cfg),
            "split_shift": manifest["split_shift"],
            "prefix_generations": prefix,
            "challenge_generations": challenge,
            "recovery_generations": recovery,
            "continuation_checks": {str(s): True for s in seeds},
            "arms": {
                name: {"environment": asdict(env), "controls": asdict(controls)}
                for name, (env, controls) in arms.items()
            },
            "provenance": (
                manifest["resume_provenance"] or [manifest["initial_provenance"]]
            )[-1],
            "uncertainty": "std is sample standard deviation across independent root seeds, not a confidence interval",
            "contrast_definition": "k-fold high-minus-low difference, averaged over remaining factors",
        }
    )
    summary = read_json(root / "summary.json")

    # The numeric tolerance covers arithmetic/library roundoff only, not IDs or counts.
    def same(a, b):
        import math

        if type(a) is float and type(b) is float:
            return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    if not same(summary, computed):
        raise ValueError("summary differs from recomputed complete paired evidence")
    return {
        "status": "PASS",
        "root_seeds": len(seeds),
        "arms_per_root": len(arms),
        "individual_records": records,
        "bound_checkpoints": len(seeds) * (1 + 2 * len(arms)),
        "hashed_files": len(expected),
        "source_matches_current": manifest["initial_provenance"]["source_sha256"]
        == execution_metadata()["source_sha256"],
        "scope": "Recomputed arithmetic, selection and mutation accounting; checkpoint/summary consistency. Does not rerun neural training or authenticate the evidence producer.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
