from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvironmentSpec:
    train_shift_fraction: float
    eval_shift_fraction: float
    cheap_train_accuracy: float
    cheap_eval_accuracy: float
    verifier_cost: float
    label: str


def make_environment_schedule(regime: str, generation: int) -> EnvironmentSpec:
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
    flip = torch.rand(sign.shape[0], generator=generator) > accuracy
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

    generator = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=generator).float()
    sign = y * 2.0 - 1.0
    robust = sign + robust_sigma * torch.randn(n, generator=generator)

    if conflict:
        shortcut = -sign + 0.15 * torch.randn(n, generator=generator)
        cheap = -sign + 0.15 * torch.randn(n, generator=generator)
    else:
        shifted = torch.rand(n, generator=generator) < shift_fraction
        normal_flip = torch.rand(n, generator=generator) > shortcut_accuracy
        shifted_flip = torch.rand(n, generator=generator) < shortcut_accuracy
        flip = torch.where(shifted, shifted_flip, normal_flip)
        shortcut_sign = torch.where(flip, -sign, sign)
        shortcut = shortcut_sign + 0.15 * torch.randn(n, generator=generator)
        cheap = _correlated_signal(sign, cheap_accuracy, generator)

    nuisance_1 = torch.randn(n, generator=generator)
    nuisance_2 = torch.randn(n, generator=generator)
    x = torch.stack([robust, shortcut, cheap, nuisance_1, nuisance_2], dim=1)
    return x, y
