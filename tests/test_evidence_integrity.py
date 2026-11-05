from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
import os
import random
import pytest
import torch
from alignment_selection_dynamics.causal import (
    initial_checkpoint,
    save_checkpoint,
    load_checkpoint,
    advance,
    checkpoint_fingerprint,
    Controls,
)
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json, write_bytes
from alignment_selection_dynamics.integrity import strict_json
from experiments.causal_factorial import (
    BASELINE,
    factorial_conditions,
    study_arms,
    run,
    read_trajectory,
    summarize_factorial,
    uncertainty,
)
from experiments.verify_factorial import verify


def cfg():
    return EvolutionConfig(
        population_size=3,
        parent_pool=2,
        elites=1,
        generations=3,
        life_epochs=1,
        train_examples=8,
        eval_examples=8,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "shape",
        "nonfinite",
        "trainability",
        "modes",
        "mode-alias",
        "unknown-parameter",
        "gaussian-nan",
        "seed-bool",
        "generation-bool",
        "mutation",
        "config-alias",
    ],
)
def test_checkpoint_corruption_rejected_without_changing_rng(tmp_path, damage):
    state, _, _ = advance(initial_checkpoint(73, cfg()), [BASELINE], cfg())
    path = tmp_path / "x.pt"
    save_checkpoint(path, state)
    data = torch.load(path, weights_only=True)
    if damage == "shape":
        data["agents"][0]["weights"]["robust_head.weight"] = torch.ones(1)
    if damage == "nonfinite":
        data["agents"][0]["weights"]["robust_head.weight"].fill_(float("inf"))
    if damage == "trainability":
        data["agents"][0]["requires_grad"]["cheap_head.weight"] = 1
    if damage == "modes":
        data["agents"][0]["modes"].pop("robust_head")
    if damage == "mode-alias":
        data["agents"][0]["modes"]["robust_head"] = 1
    if damage == "unknown-parameter":
        data["agents"][0]["weights"]["surprise"] = torch.ones(1)
    if damage == "gaussian-nan":
        data["parent_state"] = (*data["parent_state"][:2], float("nan"))
    if damage == "seed-bool":
        data["seed"] = True
    if damage == "generation-bool":
        data["next_generation"] = True
    if damage == "mutation":
        data["births"][1]["mutation"]["weights"] = {}
    if damage == "config-alias":
        data["config"]["elites"] = True
    torch.save(data, path)
    before = torch.get_rng_state().clone()
    py = random.getstate()
    with pytest.raises(ValueError):
        load_checkpoint(path)
    assert torch.equal(before, torch.get_rng_state()) and py == random.getstate()


def test_legacy_checkpoint_infers_uniform_module_modes(tmp_path):
    state = initial_checkpoint(7, cfg())
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, state)
    data = torch.load(path, weights_only=True)
    data.pop("schema_version")
    for item in data["agents"]:
        item.pop("modes")
    torch.save(data, path)
    assert checkpoint_fingerprint(load_checkpoint(path)) == checkpoint_fingerprint(
        state
    )


def test_malformed_state_rejected_before_global_settings_change(monkeypatch):
    state = initial_checkpoint(7, cfg())
    state.parent_state = (1, 2)
    monkeypatch.setattr(
        torch, "set_num_threads", lambda *a: pytest.fail("changed threads")
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *a: pytest.fail("changed determinism"),
    )
    with pytest.raises(ValueError):
        advance(state, [BASELINE], cfg())


def test_split_shift_arms_change_only_the_declared_distribution():
    arms = study_arms(True)
    assert len(arms) == 14
    for name, changed in [
        ("train_shift_only", "train_shift_fraction"),
        ("eval_shift_only", "eval_shift_fraction"),
    ]:
        env, controls = arms[name]
        before = asdict(BASELINE)
        after = asdict(env)
        assert {
            key for key in before if key != "label" and before[key] != after[key]
        } == {changed}
        assert controls == Controls()


@pytest.mark.parametrize("term", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_all_factorial_contrasts_match_independent_sign_design(term):
    runs = {}
    for bits, name in zip(itertools.product((0, 1), repeat=3), factorial_conditions()):
        sign = 1
        for axis in term:
            sign *= 2 * bits[axis] - 1
        value = 0.5 + 0.01 * sign
        runs[name] = [
            {
                "seed": 7,
                "history": [
                    {
                        "generation": g,
                        **{
                            k: value
                            for k in (
                                "alignment_score",
                                "capability",
                                "verifier_strength",
                                "bypass_strength",
                            )
                        },
                    }
                    for g in (10, 11)
                ],
            }
        ]
    output = summarize_factorial(runs, 1)
    target = "_x_".join(("bypass", "shift_removal", "cost")[axis] for axis in term)
    for name, result in output["contrasts"].items():
        assert result["end_challenge"]["alignment_score"]["mean"] == pytest.approx(
            0.01 * 2 ** len(term) if name == target else 0, abs=1e-14
        )
        assert result["end_challenge"]["alignment_score"]["n"] == 1


def test_descriptive_bootstrap_counts_roots_and_is_repeatable():
    result = uncertainty([-0.2, 0.1, 0.05, 0.0, -0.1])
    assert result == uncertainty([-0.2, 0.1, 0.05, 0.0, -0.1])
    assert result["n"] == 5 and len(result["paired_values"]) == 5
    assert uncertainty([0.2])["bootstrap_percentile_95"] is None


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    output = tmp_path_factory.mktemp("complete") / "study"
    run([73, 89], output, cfg(), prefix=1, challenge=1, split_shift=True)
    return output


def rehash(directory):
    completion = json.loads((directory / "completion.json").read_text())
    completion["files"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in completion["files"]
    }
    (directory / "completion.json").write_text(json.dumps(completion))


def test_complete_evidence_recomputes_all_accounting(completed):
    result = verify(completed)
    assert result["root_seeds"] == 2 and result["arms_per_root"] == 14
    assert result["individual_records"] == 174 and result["bound_checkpoints"] == 58


@pytest.mark.parametrize(
    "damage",
    [
        "summary",
        "false-count",
        "bad-rate",
        "bad-environment",
        "bad-parent",
        "bad-delta",
        "bad-rng",
        "bad-weight",
        "missing-file",
        "extra-file",
        "duplicate-seed",
        "unknown-control",
        "completion-bool",
    ],
)
def test_rehashed_evidence_cannot_hide_internal_contradictions(
    tmp_path, completed, damage
):
    import gzip, shutil

    root = tmp_path / "damaged"
    shutil.copytree(completed, root)
    trajectory = root / "seed-73/b1_s1_c1/trajectory.json.gz"
    if damage in (
        "false-count",
        "bad-rate",
        "bad-environment",
        "bad-parent",
        "bad-delta",
    ):
        payload = read_trajectory(trajectory)
        if damage == "false-count":
            payload["individuals"][0]["offspring_count"] += 1
        if damage == "bad-rate":
            payload["history"][0]["alignment_score"] = 0.123
        if damage == "bad-environment":
            payload["history"][0]["environment"]["verifier_cost"] = 0.2
        if damage == "bad-parent":
            payload["individuals"][0]["parent_id"] = "another-agent"
        if damage == "bad-delta":
            payload["individuals"][-1]["mutation"]["verifier_logit"] += 0.1
        trajectory.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    elif damage in ("bad-rng", "bad-weight"):
        path = root / "seed-73/b1_s1_c1/final.pt"
        data = torch.load(path, weights_only=True)
        if damage == "bad-rng":
            data["parent_state"] = random.Random(800).getstate()
        else:
            data["agents"][0]["weights"]["cheap_head.weight"] += 0.5
        torch.save(data, path)
    elif damage == "summary":
        path = root / "summary.json"
        data = json.loads(path.read_text())
        data["conditions"]["b1_s1_c1"]["end_challenge"]["alignment_score"]["mean"] = (
            0.123
        )
        path.write_text(json.dumps(data))
    elif damage == "missing-file":
        trajectory.unlink()
    elif damage == "extra-file":
        (root / "stray.json").write_text("{}")
    elif damage in ("duplicate-seed", "unknown-control"):
        path = root / "manifest.json"
        data = json.loads(path.read_text())
        if damage == "duplicate-seed":
            data["seeds"] = [73, 73]
        else:
            data["split_shift"] = "false"
        path.write_text(json.dumps(data))
    else:
        path = root / "completion.json"
        data = json.loads(path.read_text())
        data["schema_version"] = True
        path.write_text(json.dumps(data))
    if damage != "missing-file":
        rehash(root)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(root)


def test_resume_rejects_changed_completed_ledger(tmp_path, monkeypatch):
    import gzip
    import experiments.causal_factorial as runner

    out = tmp_path / "interrupted"
    original = runner.write_trajectory

    def stop(path, payload):
        original(path, payload)
        if path.parent.name == "b0_s0_c1":
            raise RuntimeError("pause")

    monkeypatch.setattr(runner, "write_trajectory", stop)
    with pytest.raises(RuntimeError):
        run([73], out, cfg(), prefix=1, challenge=1)
    monkeypatch.setattr(runner, "write_trajectory", original)
    path = out / "seed-73/b0_s0_c1/trajectory.json.gz"
    data = read_trajectory(path)
    data["history"][0]["alignment_score"] = 0.12345
    path.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    with pytest.raises(ValueError):
        run([73], out, cfg(), prefix=1, challenge=1, resume=True)
    assert not (out / "summary.json").exists()


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[" * 100 + "0" + "]" * 100]
)
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_atomic_write_failure_preserves_previous_bytes(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_text("original")

    def fail(*a):
        raise OSError("fault")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        write_json(target, {"new": 1})
    assert target.read_text() == "original" and len(list(tmp_path.iterdir())) == 1


def test_exclusive_writer_cannot_replace_prior_evidence(tmp_path):
    path = tmp_path / "out"
    write_bytes(path, b"original", exclusive=True)
    with pytest.raises(FileExistsError):
        write_bytes(path, b"new", exclusive=True)
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("module_name", ["selection_regimes", "shortcut_challenge"])
def test_legacy_runner_refuses_existing_record_before_training(
    tmp_path, monkeypatch, module_name
):
    import importlib

    module = importlib.import_module("experiments." + module_name)
    (tmp_path / (module_name + ".json")).write_text("previous")
    monkeypatch.setattr(
        module,
        "run_evolution",
        lambda *a: pytest.fail("trained before checking output"),
    )
    with pytest.raises(FileExistsError):
        module.run([7], tmp_path, EvolutionConfig())


def test_showcase_executes_eight_contracts(tmp_path):
    from experiments.showcase import demonstrate

    assert demonstrate(tmp_path / "showcase")["passed"] == 8


def test_installed_version_matches_runtime():
    import importlib.metadata
    import alignment_selection_dynamics as package

    assert (
        importlib.metadata.version("alignment-selection-dynamics")
        == package.__version__
    )


@pytest.mark.parametrize(
    "other,expected",
    [
        (-0.3612772269036276, True),
        (-0.3612772269036275 + 1e-7, False),
        (True, False),
        (-0.36, False),
    ],
)
def test_ancestry_rounding_is_typed_and_tightly_bounded(other, expected):
    from alignment_selection_dynamics.trajectory import same_ancestry

    a = {"id": "101:1:3", "count": 2, "mutation": {"bypass_logit": -0.3612772269036275}}
    b = deepcopy(a)
    b["mutation"]["bypass_logit"] = other
    assert same_ancestry(a, b) is expected
    b = deepcopy(a)
    b["count"] = 2.0
    assert not same_ancestry(a, b)
    b = deepcopy(a)
    b["id"] = "101:1:4"
    assert not same_ancestry(a, b)
