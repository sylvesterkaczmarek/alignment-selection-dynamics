"""Synthetic pre-repair probes. No external systems or model services."""

from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from alignment_selection_dynamics.causal import (
    initial_checkpoint,
    save_checkpoint,
    load_checkpoint,
    checkpoint_fingerprint,
    advance,
)
from alignment_selection_dynamics.evolution import (
    EvolutionConfig,
    _binary_accuracy,
    run_evolution,
)
from alignment_selection_dynamics.metrics import summarize, aggregate_final_runs
from experiments._common import argument_parser, experiment_settings
from experiments.causal_factorial import (
    BASELINE,
    factorial_conditions,
    summarize_factorial,
)


def cfg():
    return EvolutionConfig(
        population_size=2,
        parent_pool=2,
        elites=1,
        generations=3,
        life_epochs=0,
        train_examples=4,
        eval_examples=4,
    )


@pytest.mark.parametrize(
    "labels",
    [
        torch.ones((2, 1)),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.0, 2.0]),
    ],
)
def test_accuracy_rejects_invalid_ground_truth(labels):
    with pytest.raises(ValueError):
        _binary_accuracy(torch.ones(2), labels)


def test_empty_accuracy_cannot_report_nan():
    with pytest.raises(ValueError):
        _binary_accuracy(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("values", [[True, False], ["0.1", "0.5"]])
def test_numeric_summary_must_not_coerce_metadata_types(values):
    with pytest.raises(ValueError):
        summarize(values)


@pytest.mark.parametrize(
    "damage",
    [
        "typed-config",
        "invalid-intermediate",
        "out-of-range",
        "boolean-metric",
        "unknown-regime",
    ],
)
def test_historical_summary_checks_entire_run(damage):
    run = run_evolution("capability_positive", 7, cfg())
    other = deepcopy(run)
    other["seed"] = 17
    if damage == "typed-config":
        other["config"]["life_epochs"] = False
    if damage == "invalid-intermediate":
        other["history"][0]["alignment_score"] = float("nan")
    if damage == "out-of-range":
        other["history"][-1]["alignment_score"] = 1.2
    if damage == "boolean-metric":
        other["history"][-1]["alignment_score"] = True
    if damage == "unknown-regime":
        run["regime"] = other["regime"] = "not-a-regime"
    with pytest.raises(ValueError):
        aggregate_final_runs([run, other])


def cohort():
    return {
        name: [
            {
                "seed": 7,
                "history": [
                    {
                        "generation": g,
                        **{
                            m: 0.5
                            for m in (
                                "alignment_score",
                                "capability",
                                "verifier_strength",
                                "bypass_strength",
                            )
                        },
                    }
                    for g in (1, 2)
                ],
            }
        ]
        for name in factorial_conditions()
    }


@pytest.mark.parametrize(
    "damage",
    [
        "unknown-arm",
        "boolean-metric",
        "out-of-range",
        "invalid-intermediate",
        "boolean-generation",
    ],
)
def test_factorial_summary_rejects_unusable_evidence(damage):
    runs = cohort()
    if damage == "unknown-arm":
        runs["invented"] = deepcopy(runs["b0_s0_c0"])
    if damage == "boolean-metric":
        runs["b1_s1_c1"][0]["history"][0]["alignment_score"] = True
    if damage == "out-of-range":
        runs["b1_s1_c1"][0]["history"][0]["alignment_score"] = -0.1
    if damage == "invalid-intermediate":
        for arms in runs.values():
            arms[0]["history"].insert(1, deepcopy(arms[0]["history"][0]))
            arms[0]["history"][1]["generation"] = 2
            arms[0]["history"][2]["generation"] = 3
        runs["b1_s1_c1"][0]["history"][1]["alignment_score"] = float("nan")
    if damage == "boolean-generation":
        runs["b1_s1_c1"][0]["history"][0]["generation"] = True
    with pytest.raises(ValueError):
        summarize_factorial(runs, 1)


@pytest.mark.parametrize(
    "damage",
    [
        "horizon",
        "duplicate-birth",
        "parent-rng",
        "trait-rng",
        "weight-rng",
        "mixed-weight-dtype",
        "extra-field",
    ],
)
def test_checkpoint_load_rejects_invalid_contract(tmp_path, damage):
    path = tmp_path / "state.pt"
    save_checkpoint(path, initial_checkpoint(7, cfg()))
    data = torch.load(path, weights_only=True)
    if damage == "horizon":
        data["next_generation"] = 100
    if damage == "duplicate-birth":
        data["births"][1] = deepcopy(data["births"][0])
    if damage == "parent-rng":
        data["parent_state"] = (1, 2)
    if damage == "trait-rng":
        data["trait_state"] = (1, 2)
    if damage == "weight-rng":
        data["weight_state"] = torch.zeros(3, dtype=torch.uint8)
    if damage == "mixed-weight-dtype":
        data["agents"][0]["weights"]["cheap_head.weight"] = data["agents"][0][
            "weights"
        ]["cheap_head.weight"].double()
    if damage == "extra-field":
        data["unrecognised_control"] = True
    torch.save(data, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_checkpoint_restores_individual_module_modes(tmp_path):
    state = initial_checkpoint(7, cfg())
    state.population[0].train()
    state.population[0].robust_head.eval()
    before = [m.training for m in state.population[0].modules()]
    save_checkpoint(tmp_path / "mixed.pt", state)
    after = load_checkpoint(tmp_path / "mixed.pt")
    assert [m.training for m in after.population[0].modules()] == before


def test_checkpoint_fingerprint_binds_individual_module_modes():
    state = initial_checkpoint(7, cfg())
    prior = checkpoint_fingerprint(state)
    state.population[0].robust_head.eval()
    assert checkpoint_fingerprint(state) != prior


def test_advance_cannot_silently_change_fixed_configuration():
    state = initial_checkpoint(7, cfg())
    with pytest.raises(ValueError):
        advance(state, [BASELINE], replace(cfg(), learning_rate=0.5))


def test_duplicate_yaml_is_not_silently_last_wins(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("learning_rate: 0.01\nlearning_rate: 0.9\n")
    args = argument_parser("audit").parse_args(["--config", str(path)])
    with pytest.raises(ValueError):
        experiment_settings(args)
