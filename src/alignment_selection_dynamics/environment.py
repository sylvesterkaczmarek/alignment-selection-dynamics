from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


def _finite_control(name: str, value: float, *, probability: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (probability and value > 1):
        bound = "between 0 and 1" if probability else "nonnegative"
        raise ValueError(f"{name} must be {bound}")


@dataclass(frozen=True)
class EnvironmentSpec:
    train_shift_fraction: float
    eval_shift_fraction: float
    cheap_train_accuracy: float
    cheap_eval_accuracy: float
    verifier_cost: float
    label: str

    def __post_init__(self) -> None:
        for name in ("train_shift_fraction", "eval_shift_fraction", "cheap_train_accuracy", "cheap_eval_accuracy"):
            _finite_control(name, getattr(self, name), probability=True)
        _finite_control("verifier_cost", self.verifier_cost)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")


def make_environment_schedule(regime: str, generation: int) -> EnvironmentSpec:
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    if regime == "safety_only":
        return EnvironmentSpec(0.0, 0.0, 0.5, 0.5, 0.08, "safety-only")
    if regime == "neutral":
        return EnvironmentSpec(0.0, 0.0, 0.5, 0.5, 0.0, "neutral")
    if regime == "capability_positive":
        return EnvironmentSpec(0.25, 0.35, 0.5, 0.5, 0.01, "capability-positive")
    if regime == "shortcut_challenge":
        if generation < 10:
            return EnvironmentSpec(0.25, 0.35, 0.5, 0.5, 0.01, "pre-challenge")
        if generation < 20:
            return EnvironmentSpec(0.0, 0.0, 0.995, 0.995, 0.03, "cheap-shortcut")
        return EnvironmentSpec(0.25, 0.35, 0.5, 0.5, 0.01, "recovery")
    raise ValueError(f"Unknown regime: {regime}")


def _correlated_signal(
    sign: torch.Tensor,
    accuracy: float,
    generator: torch.Generator,
    noise: float = 0.15,
) -> torch.Tensor:
    flip = torch.rand(sign.shape[0], generator=generator) >= accuracy
    signal_sign = torch.where(flip, -sign, sign)
    return signal_sign + noise * torch.randn(sign.shape[0], generator=generator)


def make_dataset(
    n: int,
    seed: int,
    *,
    shortcut_accuracy: float = 0.98,
    cheap_accuracy: float = 0.5,
    robust_sigma: float = 0.7,
    shift_fraction: float = 0.0,
    conflict: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic binary task with robust, shortcut and cheap features.

    Feature columns:
      0 robust evidence
      1 shortcut evidence
      2 cheap bypass evidence
      3-4 nuisance features
    """

    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("dataset seed must be an integer in [0, 2**64)")
    for name, value in (("shortcut_accuracy", shortcut_accuracy), ("cheap_accuracy", cheap_accuracy), ("shift_fraction", shift_fraction)):
        _finite_control(name, value, probability=True)
    _finite_control("robust_sigma", robust_sigma)
    if type(conflict) is not bool:
        raise ValueError("conflict must be a boolean")

    generator = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=generator).float()
    sign = y * 2.0 - 1.0
    robust = sign + robust_sigma * torch.randn(n, generator=generator)

    if conflict:
        shortcut = -sign + 0.15 * torch.randn(n, generator=generator)
        cheap = -sign + 0.15 * torch.randn(n, generator=generator)
    else:
        shifted = torch.rand(n, generator=generator) < shift_fraction
        normal_flip = torch.rand(n, generator=generator) >= shortcut_accuracy
        shifted_flip = torch.rand(n, generator=generator) < shortcut_accuracy
        flip = torch.where(shifted, shifted_flip, normal_flip)
        shortcut_sign = torch.where(flip, -sign, sign)
        shortcut = shortcut_sign + 0.15 * torch.randn(n, generator=generator)
        cheap = _correlated_signal(sign, cheap_accuracy, generator)

    nuisance_1 = torch.randn(n, generator=generator)
    nuisance_2 = torch.randn(n, generator=generator)
    x = torch.stack([robust, shortcut, cheap, nuisance_1, nuisance_2], dim=1)
    if not torch.isfinite(x).all():
        raise ValueError("dataset generation produced non-finite features")
    return x, y
