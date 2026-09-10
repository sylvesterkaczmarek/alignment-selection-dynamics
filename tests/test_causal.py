from copy import deepcopy
from dataclasses import replace
import itertools
import random

import numpy as np
import pytest
import torch

from alignment_selection_dynamics import causal
from alignment_selection_dynamics.causal import (
    Controls, advance, checkpoint_fingerprint, initial_checkpoint, load_checkpoint,
    protocol_seed, save_checkpoint,
)
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.model import logistic
from experiments.causal_factorial import (
    BASELINE, factorial_conditions, read_trajectory, run, summarize_factorial, write_trajectory,
)


def config(**changes):
    cfg = EvolutionConfig(population_size=4, parent_pool=3, elites=1, generations=4,
                          life_epochs=1, train_examples=16, eval_examples=32)
    return replace(cfg, **changes)


@pytest.mark.parametrize("controls", [Controls(), Controls(selection="random"),
                                      Controls(mutate_traits=False), Controls(freeze="robust"),
                                      Controls(freeze="all")])
@pytest.mark.parametrize("seed", [7, 83])
def test_serialized_continuation_matches_uninterrupted_and_preserves_input(tmp_path, controls, seed):
    cfg = config()
    initial = initial_checkpoint(seed, cfg)
    original = checkpoint_fingerprint(initial)
    full, full_history, full_records = advance(initial, [BASELINE] * 4, cfg, controls)
    shared, prefix, prefix_records = advance(initial, [BASELINE] * 2, cfg, controls)
    assert shared.next_generation == 2
    path = tmp_path / "boundary.pt"
    save_checkpoint(path, shared)
    resumed = load_checkpoint(path)
    assert checkpoint_fingerprint(resumed) == checkpoint_fingerprint(shared)
    end, suffix, suffix_records = advance(resumed, [BASELINE] * 2, cfg, controls)
    assert full_history == prefix + suffix
    assert full_records == prefix_records + suffix_records
    assert checkpoint_fingerprint(end) == checkpoint_fingerprint(full)
    assert checkpoint_fingerprint(initial) == original


def test_reporting_can_be_removed_or_resized_without_changing_selection():
    cfg = config()
    state = initial_checkpoint(17, cfg)
    normal, _, normal_records = advance(state, [BASELINE] * 4, cfg)
    absent, _, absent_records = advance(state, [BASELINE] * 4, cfg, diagnostics=False)
    larger, _, larger_records = advance(state, [BASELINE] * 4, cfg, report_examples=127)
    assert checkpoint_fingerprint(normal) == checkpoint_fingerprint(absent)
    assert checkpoint_fingerprint(normal) == checkpoint_fingerprint(larger)
    keys = ("id", "parent_id", "fitness", "eligible", "offspring_count", "mutation")
    for records in (absent_records, larger_records):
        assert [[row[k] for k in keys] for row in records] == [
            [row[k] for k in keys] for row in normal_records]


def test_no_global_random_stream_is_consumed(tmp_path):
    random.seed(2)
    np.random.seed(3)
    torch.manual_seed(4)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cfg = config()
    state = initial_checkpoint(13, cfg)
    state, _, _ = advance(state, [BASELINE] * 2, cfg)
    save_checkpoint(tmp_path / "checkpoint.pt", state)
    load_checkpoint(tmp_path / "checkpoint.pt")
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_shared_fitness_is_sampled_once_and_separate_from_reporting(monkeypatch):
    calls = []
    original = causal.make_dataset
    def observed(n, seed, **kwargs):
        calls.append(seed)
        return original(n, seed, **kwargs)
    monkeypatch.setattr(causal, "make_dataset", observed)
    cfg = config(life_epochs=0)
    initial = initial_checkpoint(7, cfg)
    initial.population = [initial.population[0].clone() for _ in initial.population]
    _, history, records = advance(initial, [BASELINE], cfg)
    assert calls == [protocol_seed(7, 0, 0, name) for name in ("fitness", "report", "conflict")]
    assert len(set(calls)) == 3
    assert len({row["fitness"] for row in records}) == 1
    assert history[0]["feature_policies"]["shortcut"]["alignment_score"] == 0
    assert history[0]["feature_policies"]["cheap"]["alignment_score"] == 0


def test_random_reproduction_including_elites_ignores_every_fitness_value():
    cfg = config()
    a = initial_checkpoint(29, cfg)
    b = deepcopy(a)
    scores = [{"fitness": float(i)} for i in range(cfg.population_size)]
    reverse = [{"fitness": float(100 - i)} for i in range(cfg.population_size)]
    records_a, _ = causal._reproduce(a, scores, cfg, Controls(selection="random"))
    records_b, _ = causal._reproduce(b, reverse, cfg, Controls(selection="random"))
    assert checkpoint_fingerprint(a) == checkpoint_fingerprint(b)
    for left, right in zip(records_a, records_b):
        assert left["elite_copies"] == right["elite_copies"]
        assert left["offspring_count"] == right["offspring_count"]


def test_fixed_traits_and_recorded_mutations_reconstruct_birth_traits():
    cfg = config()
    initial = initial_checkpoint(41, cfg)
    parents = {row["id"]: agent for row, agent in zip(initial.births, initial.population)}
    end, _, _ = advance(initial, [BASELINE], cfg, Controls(mutate_traits=False))
    for birth, child in zip(end.births, end.population):
        parent = parents[birth["parent_id"]]
        assert child.traits == parent.traits
        assert birth["mutation"]["verifier_logit"] == 0
        assert birth["mutation"]["bypass_logit"] == 0
    mutated, _, _ = advance(initial, [BASELINE], cfg)
    for birth, child in zip(mutated.births, mutated.population):
        parent = parents[birth["parent_id"]]
        assert child.verifier_logit == parent.verifier_logit + birth["mutation"]["verifier_logit"]
        assert child.bypass_logit == parent.bypass_logit + birth["mutation"]["bypass_logit"]
    assert end.trait_state == mutated.trait_state
    assert torch.equal(end.weight_state, mutated.weight_state)


@pytest.mark.parametrize("freeze", ["robust", "all"])
def test_freeze_blocks_learning_and_weight_mutation(freeze):
    cfg = config()
    state = initial_checkpoint(53, cfg)
    parents = {row["id"]: agent for row, agent in zip(state.births, state.population)}
    end, _, _ = advance(state, [BASELINE], cfg, Controls(freeze=freeze))
    for birth, child in zip(end.births, end.population):
        parent = parents[birth["parent_id"]]
        for name, parameter in child.named_parameters():
            if freeze == "all" or name.startswith("robust_head."):
                assert torch.equal(parameter, dict(parent.named_parameters())[name])
                assert not parameter.requires_grad


def test_weight_mutations_are_exactly_recorded_without_learning():
    cfg = config(life_epochs=0)
    state = initial_checkpoint(53, cfg)
    parents = {row["id"]: agent for row, agent in zip(state.births, state.population)}
    end, _, _ = advance(state, [BASELINE], cfg)
    for birth, child in zip(end.births, end.population):
        parent = parents[birth["parent_id"]]
        for name, parameter in child.named_parameters():
            delta = birth["mutation"]["weights"].get(name)
            expected = dict(parent.named_parameters())[name].detach().clone()
            if delta is not None:
                expected.add_(torch.tensor(delta, dtype=expected.dtype))
            assert torch.equal(parameter, expected)


def test_realized_offspring_and_mutation_account_for_trait_change():
    cfg = config()
    state = initial_checkpoint(7, cfg)
    _, history, records = advance(state, [BASELINE] * 4, cfg)
    for row in history:
        group = [r for r in records if r["generation"] == row["generation"]]
        assert sum(r["offspring_count"] for r in group) == cfg.population_size
        assert sum(r["elite_copies"] for r in group) == cfg.elites
        assert sum(r["eligible"] for r in group) == cfg.parent_pool
        weighted = sum(r["offspring_count"] * r["verifier_strength"] for r in group) / cfg.population_size
        assert row["realized_selection_differential"] == pytest.approx(weighted - row["verifier_strength"])
        assert row["newborn_verifier_change"] == pytest.approx(
            row["realized_selection_differential"] + row["mutation_change"], abs=1e-15)


def test_logit_symmetric_mutation_is_not_sigmoid_neutral():
    assert (logistic(2 - 0.7) + logistic(2 + 0.7)) / 2 < logistic(2)
    assert (logistic(-2 - 0.7) + logistic(-2 + 0.7)) / 2 > logistic(-2)


def test_checkpoint_preserves_precision_mode_and_parameter_freezes(tmp_path):
    state = initial_checkpoint(7, config())
    state.population[0].double().eval()
    state.population[0].robust_head.weight.requires_grad_(False)
    save_checkpoint(tmp_path / "state.pt", state)
    loaded = load_checkpoint(tmp_path / "state.pt")
    assert checkpoint_fingerprint(state) == checkpoint_fingerprint(loaded)
    assert loaded.population[0].robust_head.weight.dtype == torch.float64
    assert not loaded.population[0].training
    assert not loaded.population[0].robust_head.weight.requires_grad


def test_checkpoint_and_trajectory_do_not_overwrite_existing_evidence(tmp_path):
    target = tmp_path / "state.pt"
    state = initial_checkpoint(7, config())
    save_checkpoint(target, state)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(target, state)
    assert target.read_bytes() == original
    path = tmp_path / "run.json.gz"
    write_trajectory(path, {"valid": 1})
    with pytest.raises(ValueError):
        write_trajectory(path, {"invalid": float("nan")})
    assert read_trajectory(path) == {"valid": 1}


def test_invalid_checkpoint_parameters_fail_before_creating_file(tmp_path):
    state = initial_checkpoint(7, config())
    with torch.no_grad():
        state.population[0].robust_head.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        save_checkpoint(tmp_path / "invalid.pt", state)
    assert not (tmp_path / "invalid.pt").exists()


def test_factorial_isolates_only_the_requested_controls():
    conditions = factorial_conditions()
    assert len(conditions) == 8
    for bits, env in zip(itertools.product((0, 1), repeat=3), conditions.values()):
        bypass, shift, cost = bits
        assert env.cheap_train_accuracy == env.cheap_eval_accuracy == (0.995 if bypass else 0.5)
        assert (env.train_shift_fraction, env.eval_shift_fraction) == ((0, 0) if shift else (0.25, 0.35))
        assert env.verifier_cost == (0.03 if cost else 0.01)


@pytest.mark.parametrize("seeds", [[], [7, 7], [True], [-1], [2**32]])
def test_invalid_seeds_fail_before_creating_output(seeds, tmp_path):
    with pytest.raises(ValueError):
        run(seeds, tmp_path / "invalid", config(), prefix=1, challenge=1)
    assert not (tmp_path / "invalid").exists()


def test_full_small_factorial_is_recomputable_from_individuals(tmp_path):
    cfg = config(generations=3)
    summary = run([73, 89], tmp_path / "factorial", cfg, prefix=1, challenge=1)
    cohorts = {}
    for arm in summary["arms"]:
        cohorts[arm] = []
        for seed in summary["seeds"]:
            payload = read_trajectory(tmp_path / "factorial" / f"seed-{seed}" / arm / "trajectory.json.gz")
            cohorts[arm].append({"seed": seed, "history": payload["history"]})
            for row in payload["history"]:
                group = [r for r in payload["individuals"] if r["generation"] == row["generation"]]
                for key in ("capability", "fitness", "alignment_score", "verifier_strength"):
                    assert row[key] == np.mean([r[key] for r in group])
    reproduced = summarize_factorial(cohorts, 1)
    assert reproduced["conditions"] == summary["conditions"]
    assert reproduced["contrasts"] == summary["contrasts"]
    assert summary["continuation_checks"] == {"73": True, "89": True}


def additive_cohorts():
    cohorts = {}
    for (b, s, c), name in zip(itertools.product((0, 1), repeat=3), factorial_conditions()):
        value = 0.1 + 0.1 * b + 0.2 * s + 0.1 * c + 0.2 * b * s
        cohorts[name] = [{"seed": root, "history": [
            {"generation": g, **{metric: value for metric in (
                "alignment_score", "capability", "verifier_strength", "bypass_strength")}}
            for g in (10, 11)]} for root in (7, 17)]
    return cohorts


def test_factorial_contrasts_have_correct_scale_and_root_sample_count():
    summary = summarize_factorial(additive_cohorts(), 1)
    contrasts = summary["contrasts"]
    assert contrasts["bypass"]["end_challenge"]["alignment_score"]["mean"] == pytest.approx(0.2)
    assert contrasts["bypass_x_shift_removal"]["end_challenge"]["alignment_score"]["mean"] == pytest.approx(0.2)
    assert contrasts["bypass_x_shift_removal_x_cost"]["end_challenge"]["alignment_score"]["mean"] == pytest.approx(0)
    assert contrasts["cost"]["end_challenge"]["alignment_score"]["n"] == 2


@pytest.mark.parametrize("problem", ["missing_arm", "missing_seed", "duplicate_seed", "reordered_seed",
                                     "truncated", "reordered_generations"])
def test_factorial_summary_rejects_unpaired_or_incomplete_evidence(problem):
    cohorts = additive_cohorts()
    arm = cohorts["b1_s1_c1"]
    if problem == "missing_arm":
        del cohorts["b1_s1_c1"]
    elif problem == "missing_seed":
        arm.pop()
    elif problem == "duplicate_seed":
        arm[1]["seed"] = arm[0]["seed"]
    elif problem == "reordered_seed":
        arm.reverse()
    elif problem == "truncated":
        arm[0]["history"].pop()
    else:
        arm[0]["history"].reverse()
    with pytest.raises(ValueError):
        summarize_factorial(cohorts, 1)


def test_interrupted_factorial_resumes_without_replacing_completed_evidence(tmp_path, monkeypatch):
    import hashlib
    from experiments import causal_factorial as runner
    cfg = config(generations=3)
    out = tmp_path / "interrupted"
    original = runner.write_trajectory
    def interrupt_after_first_arm(path, payload):
        original(path, payload)
        if path.parent.name == "b0_s0_c0":
            raise RuntimeError("simulated interruption after a completed comparison")
    monkeypatch.setattr(runner, "write_trajectory", interrupt_after_first_arm)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run([73], out, cfg, prefix=1, challenge=1)
    completed = out / "seed-73" / "b0_s0_c0" / "trajectory.json.gz"
    digest = hashlib.sha256(completed.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "write_trajectory", original)
    resumed = runner.run([73], out, cfg, prefix=1, challenge=1, resume=True)
    assert hashlib.sha256(completed.read_bytes()).hexdigest() == digest
    fresh = runner.run([73], tmp_path / "fresh", cfg, prefix=1, challenge=1)
    assert resumed["conditions"] == fresh["conditions"]
    assert resumed["contrasts"] == fresh["contrasts"]
    assert resumed["continuation_checks"] == {"73": True}
    with pytest.raises(FileExistsError, match="already complete"):
        runner.run([73], out, cfg, prefix=1, challenge=1, resume=True)


def test_resume_rejects_changed_settings_or_simulation_source(tmp_path, monkeypatch):
    import json
    from experiments import causal_factorial as runner
    cfg = config(generations=3)
    out = tmp_path / "factorial"
    runner.run([73], out, cfg, prefix=1, challenge=1)
    with pytest.raises(ValueError, match="settings"):
        runner.run([73], out, replace(cfg, learning_rate=0.01), prefix=1, challenge=1, resume=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["initial_provenance"]["source_sha256"]["alignment_selection_dynamics/model.py"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="simulation source"):
        runner.run([73], out, cfg, prefix=1, challenge=1, resume=True)
