from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .environment import EnvironmentSpec, make_dataset, make_environment_schedule
from .model import SelectionAgent
from .seeding import derive_data_seed, seed_everything, validate_seed


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 10
    generations: int = 30
    parent_pool: int = 5
    elites: int = 2
    life_epochs: int = 4
    train_examples: int = 256
    eval_examples: int = 512
    learning_rate: float = 0.03
    trait_mutation_std: float = 0.35
    weight_mutation_std: float = 0.01
    branch_aux_weight_robust: float = 0.25
    branch_aux_weight_shortcut: float = 0.15
    branch_aux_weight_cheap: float = 0.05

    def __post_init__(self) -> None:
        minimums = {
            "population_size": 2,
            "generations": 1,
            "parent_pool": 1,
            "elites": 0,
            "life_epochs": 0,
            "train_examples": 1,
            "eval_examples": 1,
        }
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}, excluding booleans")
        if self.parent_pool > self.population_size:
            raise ValueError("parent_pool cannot exceed population_size")
        if self.elites > self.parent_pool:
            raise ValueError("elites cannot exceed parent_pool")
        for name in (
            "learning_rate",
            "trait_mutation_std",
            "weight_mutation_std",
            "branch_aux_weight_robust",
            "branch_aux_weight_shortcut",
            "branch_aux_weight_cheap",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, excluding booleans")
            if value < 0 or (name == "learning_rate" and value == 0):
                bound = "positive" if name == "learning_rate" else "nonnegative"
                raise ValueError(f"{name} must be {bound}")


def _require_finite(tensor: torch.Tensor, description: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite {description}; refusing to report invalid experiment results")


def _train_agent(agent: SelectionAgent, env: EnvironmentSpec, seed: int, cfg: EvolutionConfig) -> None:
    x, y = make_dataset(
        cfg.train_examples,
        seed,
        shortcut_accuracy=0.98,
        cheap_accuracy=env.cheap_train_accuracy,
        shift_fraction=env.train_shift_fraction,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)

    for _ in range(cfg.life_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = agent(x)
        for name, output in outputs.items():
            _require_finite(output, f"training output {name}")
        loss = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], y)
        loss = loss + cfg.branch_aux_weight_robust * nn.functional.binary_cross_entropy_with_logits(
            outputs["robust_logit"], y
        )
        loss = loss + cfg.branch_aux_weight_shortcut * nn.functional.binary_cross_entropy_with_logits(
            outputs["shortcut_logit"], y
        )
        loss = loss + cfg.branch_aux_weight_cheap * nn.functional.binary_cross_entropy_with_logits(
            outputs["cheap_logit"], y
        )
        _require_finite(loss, "training loss")
        loss.backward()
        for parameter in agent.parameters():
            if parameter.grad is not None:
                _require_finite(parameter.grad, "training gradient")
        optimizer.step()
        for parameter in agent.parameters():
            _require_finite(parameter, "updated model parameter")


def _binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    _require_finite(logits, "evaluation logits")
    return float(((logits > 0.0) == (labels > 0.5)).float().mean().item())


def _evaluate_agent(
    agent: SelectionAgent,
    env: EnvironmentSpec,
    seed: int,
    cfg: EvolutionConfig,
    *,
    conflict_seed: int,
) -> dict[str, float]:
    x, y = make_dataset(
        cfg.eval_examples,
        seed,
        shortcut_accuracy=0.98,
        cheap_accuracy=env.cheap_eval_accuracy,
        shift_fraction=env.eval_shift_fraction,
    )
    conflict_x, conflict_y = make_dataset(cfg.eval_examples, conflict_seed, conflict=True)

    with torch.no_grad():
        outputs = agent(x)
        conflict_outputs = agent(conflict_x)
        capability = _binary_accuracy(outputs["logit"], y)
        alignment_score = _binary_accuracy(conflict_outputs["logit"], conflict_y)

    verifier_strength = agent.traits.verifier_strength
    bypass_strength = agent.traits.bypass_strength
    verification_cost = env.verifier_cost * verifier_strength
    fitness = capability - verification_cost

    return {
        "fitness": float(fitness),
        "capability": capability,
        "alignment_score": alignment_score,
        "verifier_strength": verifier_strength,
        "bypass_strength": bypass_strength,
        "verification_cost": float(verification_cost),
    }


def _new_population(
    population: list[SelectionAgent],
    scores: list[dict[str, float]],
    cfg: EvolutionConfig,
    rng: random.Random,
) -> tuple[list[SelectionAgent], float]:
    fitness = np.asarray([score["fitness"] for score in scores], dtype=float)
    verifier = np.asarray([score["verifier_strength"] for score in scores], dtype=float)
    order = np.argsort(-fitness, kind="stable")
    parent_indices = order[: cfg.parent_pool]

    selection_differential = float(verifier[parent_indices].mean() - verifier.mean())
    parents = [population[int(i)] for i in parent_indices]

    next_population = [parents[i].clone() for i in range(cfg.elites)]

    while len(next_population) < cfg.population_size:
        child = rng.choice(parents).clone()
        child.verifier_logit += rng.gauss(0.0, cfg.trait_mutation_std)
        child.bypass_logit += rng.gauss(0.0, cfg.trait_mutation_std)
        with torch.no_grad():
            for parameter in child.parameters():
                parameter.add_(cfg.weight_mutation_std * torch.randn_like(parameter))
        next_population.append(child)

    return next_population, selection_differential


def run_evolution(regime: str, seed: int, cfg: EvolutionConfig | None = None) -> dict:
    cfg = EvolutionConfig() if cfg is None else cfg
    if not isinstance(cfg, EvolutionConfig):
        raise TypeError("cfg must be an EvolutionConfig")
    validate_seed(seed)
    make_environment_schedule(regime, 0)

    seed_everything(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    population: list[SelectionAgent] = []
    for _ in range(cfg.population_size):
        population.append(
            SelectionAgent(
                verifier_logit=float(np_rng.normal(-0.3, 0.35)),
                bypass_logit=float(np_rng.normal(-2.5, 0.25)),
            )
        )

    history: list[dict] = []
    for generation in range(cfg.generations):
        env = make_environment_schedule(regime, generation)
        generation_scores: list[dict[str, float]] = []

        for index, agent in enumerate(population):
            train_seed = derive_data_seed(seed, generation, index, 0)
            fitness_seed = derive_data_seed(seed, generation, index, 1)
            conflict_seed = derive_data_seed(seed, generation, index, 2)
            _train_agent(agent, env, train_seed, cfg)
            generation_scores.append(
                _evaluate_agent(agent, env, fitness_seed, cfg, conflict_seed=conflict_seed)
            )

        next_population, selection_differential = _new_population(population, generation_scores, cfg, rng)
        numeric_keys = generation_scores[0].keys()
        means = {
            key: float(np.mean([score[key] for score in generation_scores])) for key in numeric_keys
        }
        means.update(
            {
                "generation": generation,
                "environment": env.label,
                "selection_differential": selection_differential,
                "best_fitness": float(max(score["fitness"] for score in generation_scores)),
            }
        )
        history.append(means)
        population = next_population

    return {
        "regime": regime,
        "seed": seed,
        "config": asdict(cfg),
        "history": history,
    }
