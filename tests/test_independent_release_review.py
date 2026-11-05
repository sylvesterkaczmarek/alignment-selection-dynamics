"""Independent release probes using owned synthetic studies and deliberately corrupted copies."""

from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
import shutil
import pytest
import torch
from alignment_selection_dynamics.causal import (
    initial_checkpoint,
    advance,
    load_checkpoint,
    checkpoint_fingerprint,
    Controls,
)
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.trajectory import validate_segment
from experiments import causal_factorial as runner
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


def get(path):
    return (
        json.loads(gzip.decompress(path.read_bytes()))
        if str(path).endswith(".gz")
        else json.loads(path.read_text())
    )


def put(path, data):
    raw = json.dumps(data, sort_keys=True).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw)


def rehash(root):
    data = get(root / "completion.json")
    data["files"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in data["files"]
    }
    put(root / "completion.json", data)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent") / "study"
    runner.run([1], root, cfg(), prefix=1, challenge=1, split_shift=True)
    return root


def test_positive_complete_control(small):
    proof = verify(small)
    assert proof["root_seeds"] == 1 and proof["arms_per_root"] == 14


@pytest.mark.parametrize(
    "kind",
    [
        "prefix-seed-alias",
        "prefix-extra",
        "arm-extra",
        "row-extra",
        "feature-accuracy",
        "feature-probe",
        "report-denominator",
        "probe-denominator",
    ],
)
def test_rehashed_ledger_contradictions_are_rejected(small, tmp_path, kind):
    root = tmp_path / "copy"
    shutil.copytree(small, root)
    prefix = kind.startswith("prefix")
    path = root / (
        "seed-1/prefix.json.gz" if prefix else "seed-1/b1_s0_c0/trajectory.json.gz"
    )
    data = get(path)
    if kind == "prefix-seed-alias":
        data["seed"] = True
    elif kind in ("prefix-extra", "arm-extra"):
        data["unexpected_samples"] = 999
    elif kind == "row-extra":
        data["history"][0]["sample_count"] = 999
    elif kind == "feature-accuracy":
        data["history"][0]["feature_policies"]["robust"]["capability"] = 0.123
    elif kind == "feature-probe":
        data["history"][0]["feature_policies"]["shortcut"]["alignment_score"] = 1.0
    else:
        metric = "capability" if kind == "report-denominator" else "alignment_score"
        # A proportion of correct labels from exactly 8 examples cannot be 0.1.
        old = data["individuals"][0][metric]
        data["individuals"][0][metric] = 0.1
        row = data["history"][0]
        row[metric] += (0.1 - old) / cfg().population_size
        summary = get(root / "summary.json")
        cohorts = {}
        for arm in summary["arms"]:
            record = (
                data
                if arm == "b1_s0_c0"
                else get(root / f"seed-1/{arm}/trajectory.json.gz")
            )
            cohorts[arm] = [{"seed": 1, "history": record["history"]}]
        stats = runner.enrich_summary(
            runner.summarize_factorial(cohorts, 1), cohorts, 1
        )
        summary.update(stats)
        put(root / "summary.json", summary)
    put(path, data)
    rehash(root)
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize(
    "kind", ["config", "controls", "mode", "trainability", "dtype"]
)
def test_segment_rejects_boundary_state_mismatch(kind):
    settings = cfg()
    initial = initial_checkpoint(73, settings)
    final, history, records = advance(initial, [runner.BASELINE], settings)
    bad = deepcopy(final)
    if kind == "config":
        bad.config["learning_rate"] = 0.2
    if kind == "controls":
        bad.controls["selection"] = "random"
    if kind == "mode":
        bad.population[0].robust_head.eval()
    if kind == "trainability":
        bad.population[0].robust_head.weight.requires_grad_(False)
    if kind == "dtype":
        bad.population[0].double()
    with pytest.raises(ValueError):
        validate_segment(
            history, records, initial, bad, settings, Controls(), [runner.BASELINE]
        )


@pytest.mark.parametrize("kind", ["shape", "frozen-weights"])
def test_no_learning_segment_rejects_changed_parameters(kind):
    settings = replace(cfg(), life_epochs=0)
    start = initial_checkpoint(73, settings)
    controls = Controls(freeze="all")
    end, history, records = advance(start, [runner.BASELINE], settings, controls)
    if kind == "shape":
        end.population[0].robust_head.weight = torch.nn.Parameter(torch.ones(1, 4))
    else:
        with torch.no_grad():
            end.population[0].robust_head.weight.add_(1)
    with pytest.raises(ValueError):
        validate_segment(
            history, records, start, end, settings, controls, [runner.BASELINE]
        )


@pytest.mark.parametrize(
    "kind",
    [
        "checkpoint-config",
        "checkpoint-control",
        "record-config-alias",
        "record-control-alias",
        "protocol",
    ],
)
def test_resume_rejects_mismatched_completed_arm(tmp_path, monkeypatch, kind):
    root = tmp_path / "interrupted"
    original = runner.write_trajectory

    def interrupt(path, data):
        original(path, data)
        if path.parent.name == "b0_s0_c1":
            raise RuntimeError("synthetic pause")

    monkeypatch.setattr(runner, "write_trajectory", interrupt)
    with pytest.raises(RuntimeError, match="synthetic pause"):
        runner.run([1], root, cfg(), prefix=1, challenge=1)
    monkeypatch.setattr(runner, "write_trajectory", original)
    path = root / "seed-1/b0_s0_c1/trajectory.json.gz"
    payload = get(path)
    if kind.startswith("checkpoint"):
        checkpoint_path = path.parent / "final.pt"
        data = torch.load(checkpoint_path, weights_only=True)
        if kind == "checkpoint-config":
            data["config"]["learning_rate"] = 0.2
        else:
            data["controls"]["selection"] = "random"
        torch.save(data, checkpoint_path)
        payload["final_checkpoint_fingerprint"] = checkpoint_fingerprint(
            load_checkpoint(checkpoint_path)
        )
    elif kind == "record-config-alias":
        payload["config"]["elites"] = True
    elif kind == "record-control-alias":
        payload["controls"]["mutate_traits"] = 1
    else:
        payload["protocol"] = "unrelated"
    put(path, payload)
    with pytest.raises(ValueError):
        runner.run([1], root, cfg(), prefix=1, challenge=1, resume=True)
    assert not (root / "completion.json").exists()


@pytest.mark.parametrize(
    "kind", ["tuple-roots", "source-set", "source-hash", "runtime-shape"]
)
def test_manifest_rejects_malformed_provenance(tmp_path, small, kind):
    root = tmp_path / "copy"
    shutil.copytree(small, root)
    path = root / "manifest.json"
    data = get(path)
    if kind == "tuple-roots":
        data["seeds"] = {"1": 1}
    elif kind == "source-set":
        data["initial_provenance"]["source_sha256"] = {}
    elif kind == "source-hash":
        data["initial_provenance"]["source_sha256"][
            next(iter(data["initial_provenance"]["source_sha256"]))
        ] = "not-a-sha256"
    else:
        data["initial_provenance"]["torch_settings"]["num_threads"] = True
    summary = get(root / "summary.json")
    summary["provenance"] = data["initial_provenance"]
    put(root / "summary.json", summary)
    put(path, data)
    rehash(root)
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize(
    "controls",
    [
        Controls(),
        Controls(selection="random"),
        Controls(mutate_traits=False),
        Controls(freeze="robust"),
        Controls(freeze="all"),
    ],
)
def test_valid_declared_segment_controls_remain_supported(controls):
    settings = cfg()
    start = initial_checkpoint(73, settings)
    end, history, records = advance(start, [runner.BASELINE] * 3, settings, controls)
    assert (
        validate_segment(
            history, records, start, end, settings, controls, [runner.BASELINE] * 3
        )["individual_records"]
        == 9
    )


@pytest.mark.parametrize(
    "controls",
    [
        Controls(),
        Controls(selection="random"),
        Controls(mutate_traits=False),
        Controls(freeze="robust"),
        Controls(freeze="all"),
    ],
)
def test_mixed_modes_and_double_precision_survive_valid_lineage(controls):
    settings = cfg()
    start = initial_checkpoint(17, settings)
    for i, agent in enumerate(start.population):
        agent.double()
        agent.train(i != 0)
        agent.robust_head.train(i == 0)
    end, history, records = advance(start, [runner.BASELINE] * 3, settings, controls)
    assert (
        validate_segment(
            history, records, start, end, settings, controls, [runner.BASELINE] * 3
        )["individual_records"]
        == 9
    )


@pytest.mark.parametrize("count", [3, 17, 31])
def test_explicit_reporting_sample_count_is_verified(count):
    settings = cfg()
    start = initial_checkpoint(31, settings)
    end, history, records = advance(
        start, [runner.BASELINE], settings, report_examples=count
    )
    assert (
        validate_segment(
            history,
            records,
            start,
            end,
            settings,
            Controls(),
            [runner.BASELINE],
            report_examples=count,
        )["generations"]
        == 1
    )


@pytest.mark.parametrize("frozen", ["robust", "all"])
def test_frozen_boundary_tensors_are_bound_to_their_ancestors(frozen):
    settings = cfg()
    start = initial_checkpoint(17, settings)
    controls = Controls(freeze=frozen)
    end, history, records = advance(start, [runner.BASELINE] * 3, settings, controls)
    with torch.no_grad():
        end.population[1].robust_head.bias.add_(0.01)
    with pytest.raises(ValueError):
        validate_segment(
            history, records, start, end, settings, controls, [runner.BASELINE] * 3
        )


def test_checkpoint_seed_cannot_change_mid_segment():
    settings = cfg()
    start = initial_checkpoint(17, settings)
    end, history, records = advance(start, [runner.BASELINE], settings)
    end.seed = 18
    with pytest.raises(ValueError):
        validate_segment(
            history, records, start, end, settings, Controls(), [runner.BASELINE]
        )


@pytest.mark.parametrize("flag", [True, 1.0, "8"])
def test_reporting_sample_count_cannot_be_a_typed_alias(flag):
    settings = cfg()
    start = initial_checkpoint(17, settings)
    end, history, records = advance(start, [runner.BASELINE], settings)
    with pytest.raises(ValueError):
        validate_segment(
            history,
            records,
            start,
            end,
            settings,
            Controls(),
            [runner.BASELINE],
            report_examples=flag,
        )


def test_altered_data_seed_is_not_a_new_report_control():
    settings = cfg()
    start = initial_checkpoint(17, settings)
    end, history, records = advance(start, [runner.BASELINE], settings)
    history[0]["data_seeds"]["report"] += 1
    with pytest.raises(ValueError):
        validate_segment(
            history, records, start, end, settings, Controls(), [runner.BASELINE]
        )
