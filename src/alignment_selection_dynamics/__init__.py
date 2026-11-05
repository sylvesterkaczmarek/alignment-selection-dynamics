"""Alignment Selection Dynamics research package."""

from .evolution import EvolutionConfig, run_evolution
from .environment import EnvironmentSpec, make_environment_schedule

__all__ = ["EvolutionConfig", "EnvironmentSpec", "make_environment_schedule", "run_evolution"]

__version__ = "0.2.0rc1"
