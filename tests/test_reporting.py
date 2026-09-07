from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import aggregate_final_runs, summarize
from alignment_selection_dynamics.provenance import execution_metadata
from experiments import selection_regimes, shortcut_challenge
from experiments._common import argument_parser, experiment_settings


def cohort() -> list[dict]:
    cfg = EvolutionConfig(generations=30)
    keys = ("fitness", "capability", "alignment_score", "verifier_strength",
            "bypass_strength", "selection_differential")
    return [
        {
            "seed": seed,
            "regime": "shortcut_challenge",
            "config": asdict(cfg),
            "history": [
                {"generation": generation, **{key: generation / 100 + offset for key in keys}}
                for generation in range(cfg.generations)
            ],
        }
        for seed, offset in ((7, 0.0), (17, 0.1))
    ]


@pytest.mark.parametrize("values", ([float("nan")], [float("inf")], [-float("inf")], [[1, 2]], [1e308, 1e308]))
def test_summary_rejects_invalid_or_overflowed_statistics(values):
    with pytest.raises(ValueError):
        summarize(values)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_json_failure_preserves_existing_artifact(tmp_path, bad_value):
    target = tmp_path / "result.json"
    target.write_text('{"previous": true}\n')
    with pytest.raises(ValueError):
        write_json(target, {"metric": bad_value})
    assert json.loads(target.read_text()) == {"previous": True}


@pytest.mark.parametrize("module", [selection_regimes, shortcut_challenge])
@pytest.mark.parametrize("seeds", [[], [7, 7], [True], [-1], [2**32], [3.5]])
def test_invalid_seed_cohort_fails_before_any_training(module, seeds, monkeypatch, tmp_path):
    def unexpected_training(*args, **kwargs):
        pytest.fail("Invalid cohorts must be rejected before running evolution")
    monkeypatch.setattr(module, "run_evolution", unexpected_training)
    with pytest.raises(ValueError):
        module.run(seeds, tmp_path, EvolutionConfig())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("generations", [1, 19, 20, 29])
def test_incomplete_challenge_fails_before_training(generations, monkeypatch, tmp_path):
    def unexpected_training(*args, **kwargs):
        pytest.fail("Incomplete challenge must be rejected before running evolution")
    monkeypatch.setattr(shortcut_challenge, "run_evolution", unexpected_training)
    with pytest.raises(ValueError, match="at least 30"):
        shortcut_challenge.run([7], tmp_path, EvolutionConfig(generations=generations))


def test_phase_snapshot_uses_requested_generation_and_independent_seeds():
    runs = cohort()
    snapshot = shortcut_challenge.phase_snapshot(runs, 19)
    assert snapshot["capability"]["mean"] == pytest.approx(0.24)
    assert snapshot["capability"]["std"] == pytest.approx(0.1 / 2**0.5)
    assert snapshot["capability"]["n"] == 2
    assert aggregate_final_runs(runs)["capability"]["mean"] == pytest.approx(0.34)


@pytest.mark.parametrize("problem", ["duplicate", "regime", "configuration", "truncated", "reordered"])
@pytest.mark.parametrize("consumer", [aggregate_final_runs, lambda runs: shortcut_challenge.phase_snapshot(runs, 19)])
def test_aggregation_rejects_incompatible_histories(problem, consumer):
    runs = cohort()
    if problem == "duplicate":
        runs[1]["seed"] = runs[0]["seed"]
    elif problem == "regime":
        runs[1]["regime"] = "neutral"
    elif problem == "configuration":
        runs[1]["config"]["learning_rate"] = 0.1
    elif problem == "truncated":
        runs[1]["history"].pop()
    elif problem == "reordered":
        runs[1]["history"][19], runs[1]["history"][20] = runs[1]["history"][20], runs[1]["history"][19]
    with pytest.raises(ValueError):
        consumer(runs)


@pytest.mark.parametrize("generation", [-1, True, 30, 3.5])
def test_phase_snapshot_rejects_invalid_generation(generation):
    with pytest.raises(ValueError):
        shortcut_challenge.phase_snapshot(cohort(), generation)


def test_phase_snapshot_rejects_other_experiment():
    runs = cohort()
    for run in runs:
        run["regime"] = "neutral"
    with pytest.raises(ValueError, match="shortcut_challenge"):
        shortcut_challenge.phase_snapshot(runs, 19)


def test_yaml_configuration_and_explicit_overrides_are_applied(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seeds: [3, 5]\ngenerations: 40\npopulation_size: 8\nbranch_aux_weight_robust: 0.4\nlearning_rate: 0.02\n")
    parser = argument_parser("test")
    seeds, cfg = experiment_settings(parser.parse_args(["--config", str(path)]))
    assert seeds == [3, 5]
    assert cfg.generations == 40
    assert cfg.population_size == 8
    assert cfg.branch_aux_weight_robust == 0.4
    assert cfg.learning_rate == 0.02
    seeds, cfg = experiment_settings(parser.parse_args([
        "--config", str(path), "--seeds", "11", "13", "--generations", "32", "--population", "6",
    ]))
    assert seeds == [11, 13]
    assert cfg.generations == 32
    assert cfg.population_size == 6
    assert cfg.branch_aux_weight_robust == 0.4


@pytest.mark.parametrize("content", [
    "learning_rat: 0.1", "seeds: [7, 7]", "seeds: [true]", "seeds: 7",
    "generations: 1.5", "learning_rate: .nan", "population_size: true", "- 1\n- 2", "",
])
def test_yaml_configuration_rejects_invalid_settings(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    args = argument_parser("test").parse_args(["--config", str(path)])
    with pytest.raises(ValueError):
        experiment_settings(args)


def test_provenance_records_executed_source_and_runtime():
    from alignment_selection_dynamics import metrics
    import torch
    metadata = execution_metadata()
    assert metadata["python"]
    assert metadata["packages"]["torch"]
    assert metadata["device"] == "cpu"
    assert metadata["torch_settings"]["num_threads"] == torch.get_num_threads()
    assert metadata["torch_settings"]["deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()
    assert metadata["source_sha256"]["alignment_selection_dynamics/metrics.py"] == hashlib.sha256(
        Path(metrics.__file__).read_bytes()
    ).hexdigest()
    assert metadata["source_sha256"]["experiments/selection_regimes.py"] == hashlib.sha256(
        Path(selection_regimes.__file__).read_bytes()
    ).hexdigest()


def test_small_selection_experiment_writes_recomputable_results(tmp_path):
    cfg = EvolutionConfig(
        generations=1, population_size=2, parent_pool=2, elites=1,
        life_epochs=1, train_examples=8, eval_examples=8,
    )
    returned = selection_regimes.run([7, 17], tmp_path, cfg)
    payload = json.loads((tmp_path / "selection_regimes.json").read_text())
    summary = json.loads((tmp_path / "selection_regimes_summary.json").read_text())
    assert payload["config"] == asdict(cfg)
    assert payload["provenance"]["source_sha256"]
    assert returned == summary
    for regime, runs in payload["runs"].items():
        assert aggregate_final_runs(runs) == summary[regime]
