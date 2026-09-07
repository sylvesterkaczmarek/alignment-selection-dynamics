import random

import pytest
import torch

from alignment_selection_dynamics.evolution import (
    EvolutionConfig,
    _binary_accuracy,
    _new_population,
    run_evolution,
)
from alignment_selection_dynamics.model import SelectionAgent


def small_config() -> EvolutionConfig:
    return EvolutionConfig(
        population_size=6,
        generations=4,
        parent_pool=3,
        elites=1,
        life_epochs=2,
        train_examples=96,
        eval_examples=128,
    )


def test_evolution_is_repeatable_for_same_seed():
    cfg = small_config()
    run_a = run_evolution("capability_positive", 11, cfg)
    run_b = run_evolution("capability_positive", 11, cfg)
    assert run_a == run_b


def test_history_has_expected_metrics():
    run = run_evolution("safety_only", 13, small_config())
    row = run["history"][-1]
    for key in [
        "fitness",
        "capability",
        "alignment_score",
        "verifier_strength",
        "bypass_strength",
        "selection_differential",
    ]:
        assert key in row


@pytest.mark.parametrize(
    "name,value",
    [
        ("population_size", 1),
        ("generations", 0),
        ("parent_pool", 0),
        ("elites", -1),
        ("life_epochs", -1),
        ("train_examples", 0),
        ("eval_examples", 0),
    ],
)
def test_invalid_counts_are_rejected_at_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        EvolutionConfig(**{name: value})


@pytest.mark.parametrize(
    "name",
    ["population_size", "generations", "parent_pool", "elites", "life_epochs", "train_examples", "eval_examples"],
)
@pytest.mark.parametrize("value", [True, 2.5, "3"])
def test_counts_are_not_silently_coerced(name, value):
    with pytest.raises(ValueError, match=name):
        EvolutionConfig(**{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "learning_rate",
        "trait_mutation_std",
        "weight_mutation_std",
        "branch_aux_weight_robust",
        "branch_aux_weight_shortcut",
        "branch_aux_weight_cheap",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True])
def test_invalid_optimisation_values_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        EvolutionConfig(**{name: value})


def test_inconsistent_selection_counts_are_rejected():
    with pytest.raises(ValueError, match="parent_pool"):
        EvolutionConfig(population_size=3, parent_pool=4)
    with pytest.raises(ValueError, match="elites"):
        EvolutionConfig(parent_pool=1, elites=2)
    with pytest.raises(ValueError, match="learning_rate"):
        EvolutionConfig(learning_rate=0)


def test_zero_mutation_and_no_lifetime_learning_are_valid_controls():
    cfg = EvolutionConfig(
        life_epochs=0,
        elites=0,
        trait_mutation_std=0,
        weight_mutation_std=0,
        branch_aux_weight_robust=0,
        branch_aux_weight_shortcut=0,
        branch_aux_weight_cheap=0,
    )
    assert cfg.life_epochs == cfg.trait_mutation_std == cfg.weight_mutation_std == 0


class LastParentRandom(random.Random):
    def choice(self, sequence):
        return sequence[-1]


def test_fifth_parent_can_reproduce_and_pool_differential_matches_pool():
    population = [SelectionAgent(verifier_logit=i) for i in range(6)]
    scores = [
        {"fitness": 6 - i, "verifier_strength": agent.traits.verifier_strength}
        for i, agent in enumerate(population)
    ]
    cfg = EvolutionConfig(
        population_size=6, parent_pool=5, elites=1, trait_mutation_std=0, weight_mutation_std=0
    )
    offspring, differential = _new_population(population, scores, cfg, LastParentRandom(1))
    assert [agent.verifier_logit for agent in offspring] == [0, 4, 4, 4, 4, 4]
    assert differential == pytest.approx(
        sum(score["verifier_strength"] for score in scores[:5]) / 5
        - sum(score["verifier_strength"] for score in scores) / 6
    )


def test_fitness_ties_keep_original_population_order():
    population = [SelectionAgent(verifier_logit=i) for i in range(20)]
    scores = [
        {"fitness": 1 if i % 2 == 0 else 0, "verifier_strength": agent.traits.verifier_strength}
        for i, agent in enumerate(population)
    ]
    cfg = EvolutionConfig(population_size=20, parent_pool=5, elites=5)
    offspring, _ = _new_population(population, scores, cfg, random.Random(1))
    assert [agent.verifier_logit for agent in offspring[:5]] == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_cannot_be_reported_as_accuracy(value):
    with pytest.raises(FloatingPointError, match="evaluation logits"):
        _binary_accuracy(torch.tensor([value, value]), torch.tensor([0.0, 1.0]))


def test_overflowed_model_does_not_return_plausible_run_metrics():
    cfg = EvolutionConfig(
        population_size=3,
        parent_pool=2,
        elites=0,
        generations=2,
        train_examples=8,
        eval_examples=8,
        life_epochs=1,
        weight_mutation_std=1e39,
    )
    with pytest.raises(FloatingPointError, match="training output"):
        run_evolution("neutral", 1, cfg)


def test_overflowed_loss_stops_before_optimisation():
    cfg = EvolutionConfig(
        population_size=2,
        parent_pool=1,
        elites=0,
        generations=1,
        train_examples=8,
        eval_examples=8,
        life_epochs=1,
        branch_aux_weight_robust=1e308,
    )
    with pytest.raises(FloatingPointError, match="training loss"):
        run_evolution("neutral", 1, cfg)
