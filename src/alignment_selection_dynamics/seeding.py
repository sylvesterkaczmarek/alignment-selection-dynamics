from __future__ import annotations

import random

import numpy as np
import torch


def validate_seed(seed: int) -> None:
    """Validate a root seed accepted by Python, NumPy and PyTorch."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32), excluding booleans")


def derive_data_seed(seed: int, generation: int, agent: int, stream: int) -> int:
    """Derive a Torch seed from explicit coordinates instead of overlapping offsets.

    Streams 0, 1 and 2 identify training, fitness evaluation and the conflict probe.
    The 64-bit output avoids the previous fixed population/generation stride limits.
    """
    validate_seed(seed)
    for name, value in (("generation", generation), ("agent", agent)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(stream) is not int or stream not in (0, 1, 2):
        raise ValueError("stream must be 0 (train), 1 (fitness) or 2 (conflict)")
    sequence = np.random.SeedSequence([seed, generation, agent, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def seed_everything(seed: int) -> None:
    validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
