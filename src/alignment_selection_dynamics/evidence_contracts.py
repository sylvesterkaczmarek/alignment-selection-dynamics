"""Shared structural contracts for study verification and interrupted-run recovery."""

from dataclasses import asdict
import re
from .integrity import canonical, validate_tree
from .metrics import validate_seeds


def validate_provenance(value):
    fields = {
        "python",
        "platform",
        "device",
        "torch_settings",
        "packages",
        "source_sha256",
    }
    if type(value) is not dict or value.keys() != fields:
        raise ValueError("invalid execution provenance fields")
    validate_tree(value)
    if (
        any(
            type(value[k]) is not str or not value[k].strip()
            for k in ("python", "platform")
        )
        or value["device"] != "cpu"
    ):
        raise ValueError("invalid execution provenance runtime")
    settings = value["torch_settings"]
    if type(settings) is not dict or settings.keys() != {
        "num_threads",
        "num_interop_threads",
        "deterministic_algorithms",
        "deterministic_warn_only",
    }:
        raise ValueError("invalid recorded Torch settings")
    if any(
        type(settings[k]) is not int or settings[k] < 1
        for k in ("num_threads", "num_interop_threads")
    ) or any(
        type(settings[k]) is not bool
        for k in ("deterministic_algorithms", "deterministic_warn_only")
    ):
        raise ValueError("invalid typed Torch settings")
    packages = value["packages"]
    if (
        type(packages) is not dict
        or packages.keys() != {"torch", "numpy", "matplotlib", "PyYAML"}
        or any(type(v) is not str or not v.strip() for v in packages.values())
    ):
        raise ValueError("invalid runtime package versions")
    hashes = value["source_sha256"]
    required = {
        "alignment_selection_dynamics/" + n + ".py"
        for n in ("causal", "environment", "evolution", "model", "seeding")
    }
    required.add("experiments/causal_factorial.py")
    if (
        type(hashes) is not dict
        or not required <= hashes.keys()
        or any(
            re.fullmatch(
                r"(?:alignment_selection_dynamics|experiments)/[A-Za-z_][A-Za-z_0-9]*\.py",
                k,
            )
            is None
            or type(v) is not str
            or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for k, v in hashes.items()
        )
    ):
        raise ValueError("invalid or incomplete simulation source hashes")


def validate_manifest(manifest):
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
        or manifest["protocol"] != "paired-v1"
    ):
        raise ValueError("unsupported study manifest")
    if type(manifest["seeds"]) is not list:
        raise ValueError("study roots must be a list")
    validate_seeds(manifest["seeds"])
    validate_provenance(manifest["initial_provenance"])
    if type(manifest["resume_provenance"]) is not list:
        raise ValueError("resume provenance must be a list")
    for item in manifest["resume_provenance"]:
        validate_provenance(item)
        for key in (
            "python",
            "platform",
            "device",
            "packages",
            "source_sha256",
            "torch_settings",
        ):
            if canonical(item[key]) != canonical(manifest["initial_provenance"][key]):
                raise ValueError(
                    "resumed experiment changed its recorded execution contract"
                )


def validate_prefix(payload, seed, cfg, fingerprint):
    fields = {"seed", "config", "history", "individuals", "checkpoint_fingerprint"}
    if (
        type(payload) is not dict
        or payload.keys() != fields
        or type(payload["seed"]) is not int
        or payload["seed"] != seed
        or canonical(payload["config"]) != canonical(asdict(cfg))
        or payload["checkpoint_fingerprint"] != fingerprint
    ):
        raise ValueError("prefix identity or fields mismatch")


def validate_arm(payload, seed, arm, cfg, controls, prefix_fingerprint):
    fields = {
        "protocol",
        "seed",
        "arm",
        "config",
        "controls",
        "prefix_checkpoint_fingerprint",
        "challenge_checkpoint_fingerprint",
        "final_checkpoint_fingerprint",
        "history",
        "individuals",
    }
    if (
        type(payload) is not dict
        or payload.keys() != fields
        or payload["protocol"] != "paired-v1"
        or type(payload["seed"]) is not int
        or payload["seed"] != seed
        or payload["arm"] != arm
        or canonical(payload["config"]) != canonical(asdict(cfg))
        or canonical(payload["controls"]) != canonical(asdict(controls))
        or payload["prefix_checkpoint_fingerprint"] != prefix_fingerprint
    ):
        raise ValueError("arm identity or fields mismatch")
