from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
from pathlib import Path

import torch


def execution_metadata() -> dict:
    """Record the runtime and exact source bytes used to produce an output."""
    source_hashes = {}
    roots = {"alignment_selection_dynamics": Path(__file__).parent}
    experiments = importlib.util.find_spec("experiments")
    if experiments is not None and experiments.origin is not None:
        roots["experiments"] = Path(experiments.origin).parent
    for package, directory in roots.items():
        for path in sorted(directory.glob("*.py")):
            source_hashes[f"{package}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "torch_settings": {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "matplotlib", "PyYAML")
        },
        "source_sha256": source_hashes,
    }
