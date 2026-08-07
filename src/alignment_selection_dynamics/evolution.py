from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .environment import EnvironmentSpec, make_dataset, make_environment_schedule
from .model import SelectionAgent
from .seeding import seed_everything


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
        loss.backward()
        optimizer.step()


def _binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(((logits > 0.0) == (labels > 0.5)).float().mean().item())


def _evaluate_agent(agent: SelectionAgent, env: EnvironmentSpec, seed: int, cfg: EvolutionConfig) -> dict[str, float]:
    x, y = make_dataset(
        cfg.eval_examples,
        seed,
        shortcut_accuracy=0.98,
        cheap_accuracy=env.cheap_eval_accuracy,
        shift_fraction=env.eval_shift_fraction,
    )
    conflict_x, conflict_y = make_dataset(cfg.eval_examples, seed + 10_003, conflict=True)

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
    order = np.argsort(-fitness)
    parent_indices = order[: cfg.parent_pool]

    selection_differential = float(verifier[parent_indices].mean() - verifier.mean())
    parents = [population[int(i)] for i in parent_indices]

    next_population = [parents[i].clone() for i in range(min(cfg.elites, len(parents)))]
    selectable_parents = parents[: max(1, min(4, len(parents)))]

    while len(next_population) < cfg.population_size:
        child = rng.choice(selectable_parents).clone()
        child.verifier_logit += rng.gauss(0.0, cfg.trait_mutation_std)
        child.bypass_logit += rng.gauss(0.0, cfg.trait_mutation_std)
        with torch.no_grad():
            for parameter in child.parameters():
                parameter.add_(cfg.weight_mutation_std * torch.randn_like(parameter))
        next_population.append(child)

    return next_population, selection_differential


def run_evolution(regime: str, seed: int, cfg: EvolutionConfig | None = None) -> dict:
    cfg = cfg or EvolutionConfig()
    if cfg.population_size < 2:
        raise ValueError("population_size must be at least 2")
    if cfg.parent_pool > cfg.population_size:
        raise ValueError("parent_pool cannot exceed population_size")

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
            base_seed = seed * 100_000 + generation * 1_000 + index * 10
            _train_agent(agent, env, base_seed, cfg)
            generation_scores.append(_evaluate_agent(agent, env, base_seed + 5, cfg))

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
