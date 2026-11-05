"""Independent arithmetic and boundary-learning reference for the synthetic study.

Expected scores use direct tensor operations and saved weights, never SelectionAgent,
_train_agent, advance, validate_segment or the production summary functions. Adam and
PyTorch primitives remain shared dependencies. Execute only trusted local checkpoints.
"""

from pathlib import Path
import argparse
import gzip
import itertools
import json
import math
import random
import statistics
import numpy as np
import torch
import torch.nn.functional as F

STREAMS = (
    "train",
    "fitness",
    "report",
    "conflict",
    "initial_weights",
    "initial_traits",
    "parents",
    "trait_mutation",
    "weight_mutation",
)
METRICS = ("alignment_score", "capability", "verifier_strength", "bypass_strength")


def seed_for(root, generation, slot, stream):
    return int(
        np.random.SeedSequence(
            [0x41534431, root, generation, slot, STREAMS.index(stream)]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def data(n, seed, env, split):
    rng = torch.Generator().manual_seed(seed)
    target = torch.randint(2, (n,), generator=rng).float()
    sign = 2 * target - 1
    robust = sign + 0.7 * torch.randn(n, generator=rng)
    if split == "conflict":
        shortcut = -sign + 0.15 * torch.randn(n, generator=rng)
        cheap = -sign + 0.15 * torch.randn(n, generator=rng)
    else:
        phase = "train" if split == "train" else "eval"
        shifted = torch.rand(n, generator=rng) < env[phase + "_shift_fraction"]
        ordinary = torch.rand(n, generator=rng) >= 0.98
        reverse = torch.rand(n, generator=rng) < 0.98
        flip = torch.where(shifted, reverse, ordinary)
        shortcut = torch.where(flip, -sign, sign) + 0.15 * torch.randn(n, generator=rng)
        cheap_flip = torch.rand(n, generator=rng) >= env["cheap_" + phase + "_accuracy"]
        cheap = torch.where(cheap_flip, -sign, sign) + 0.15 * torch.randn(
            n, generator=rng
        )
    x = torch.stack(
        [
            robust,
            shortcut,
            cheap,
            torch.randn(n, generator=rng),
            torch.randn(n, generator=rng),
        ],
        1,
    )
    return x, target


def sigmoid(value):
    return (
        1 / (1 + math.exp(-value))
        if value >= 0
        else math.exp(value) / (1 + math.exp(value))
    )


def forward(weights, traits, x):
    x = x.to(next(iter(weights.values())).dtype)
    r = F.linear(
        x[:, [0, 3, 4]], weights["robust_head.weight"], weights["robust_head.bias"]
    ).flatten()
    s = F.linear(
        x[:, [1]], weights["shortcut_head.weight"], weights["shortcut_head.bias"]
    ).flatten()
    c = F.linear(
        x[:, [2]], weights["cheap_head.weight"], weights["cheap_head.bias"]
    ).flatten()
    v = sigmoid(traits["verifier_logit"])
    b = sigmoid(traits["bypass_logit"])
    correction = torch.sigmoid(-3 * r * s)
    return ((1 - b) * (s + v * correction * (r - s)) + b * c, r, s, c)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), (a, b)


def read(path):
    return (
        json.loads(gzip.decompress(path.read_bytes()))
        if path.suffix == ".gz"
        else json.loads(path.read_text())
    )


def verify(directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(directory)
    summary = read(root / "summary.json")
    manifest = read(root / "manifest.json")
    cfg = manifest["config"]
    seeds = manifest["seeds"]
    arms = summary["arms"]
    challenge = manifest["challenge"]
    cells = {}
    records_checked = 0
    fitness_checked = 0
    boundary_agents = 0
    prediction_comparisons = 0
    feature_scores = 0
    for seed in seeds:
        folder = root / f"seed-{seed}"
        pre = read(folder / "prefix.json.gz")
        payloads = [("prefix", pre)] + [
            (arm, read(folder / arm / "trajectory.json.gz")) for arm in arms
        ]
        for arm, payload in payloads:
            if arm != "prefix":
                cells[(arm, seed)] = payload["history"]
            rows = payload["individuals"]
            history = payload["history"]
            parent_rng = random.Random()
            trait_rng = random.Random()
            if arm == "prefix":
                parent_rng.seed(seed_for(seed, 0, 0, "parents"))
                trait_rng.seed(seed_for(seed, 0, 0, "trait_mutation"))
                controls = {
                    "selection": "fitness",
                    "mutate_traits": True,
                    "freeze": "none",
                }
            else:
                cp = torch.load(
                    folder / "pre_challenge.pt", map_location="cpu", weights_only=True
                )
                parent_rng.setstate(cp["parent_state"])
                trait_rng.setstate(cp["trait_state"])
                controls = arms[arm]["controls"]
            for index, row in enumerate(history):
                group = rows[
                    index * cfg["population_size"] : (index + 1)
                    * cfg["population_size"]
                ]
                assert len(group) == cfg["population_size"]
                env = row["environment"]
                generation = row["generation"]
                for item in group:
                    v = sigmoid(item["verifier_logit"])
                    b = sigmoid(item["bypass_logit"])
                    close(item["verifier_strength"], v)
                    close(item["bypass_strength"], b)
                    close(item["verification_cost"], env["verifier_cost"] * v)
                    close(
                        item["fitness"],
                        item["fitness_capability"] - env["verifier_cost"] * v,
                    )
                    fitness_checked += 1
                order = (
                    parent_rng.sample(range(len(group)), len(group))
                    if controls["selection"] == "random"
                    else sorted(range(len(group)), key=lambda i: -group[i]["fitness"])
                )
                eligible = order[: cfg["parent_pool"]]
                selected = eligible[: cfg["elites"]] + [
                    parent_rng.choice(eligible)
                    for _ in range(cfg["population_size"] - cfg["elites"])
                ]
                assert [v["offspring_count"] for v in group] == [
                    selected.count(i) for i in range(len(group))
                ]
                assert [v["eligible"] for v in group] == [
                    i in eligible for i in range(len(group))
                ]
                values = [v["verifier_strength"] for v in group]
                parent_mean = statistics.mean(values[i] for i in selected)
                close(
                    row["realized_selection_differential"],
                    parent_mean - statistics.mean(values),
                )
                close(
                    row["selection_differential"],
                    statistics.mean(values[i] for i in eligible)
                    - statistics.mean(values),
                )
                newborn = []
                for slot, parent in enumerate(selected):
                    delta_v = delta_b = 0.0
                    if slot >= cfg["elites"]:
                        drawn_v = trait_rng.gauss(0, cfg["trait_mutation_std"])
                        drawn_b = trait_rng.gauss(0, cfg["trait_mutation_std"])
                        if controls["mutate_traits"]:
                            delta_v, delta_b = drawn_v, drawn_b
                    newborn.append(sigmoid(group[parent]["verifier_logit"] + delta_v))
                    if index + 1 < len(history):
                        child = rows[(index + 1) * cfg["population_size"] + slot]
                        assert child["parent_id"] == group[parent]["id"]
                        close(
                            child["verifier_logit"],
                            group[parent]["verifier_logit"] + delta_v,
                        )
                        close(
                            child["bypass_logit"],
                            group[parent]["bypass_logit"] + delta_b,
                        )
                close(row["mutation_change"], statistics.mean(newborn) - parent_mean)
                close(
                    row["newborn_verifier_change"],
                    statistics.mean(newborn) - statistics.mean(values),
                )
                for metric in (
                    "capability",
                    "alignment_score",
                    "fitness",
                    "verifier_strength",
                    "bypass_strength",
                ):
                    close(row[metric], statistics.mean(v[metric] for v in group))
                for split, metric in [
                    ("report", "capability"),
                    ("conflict", "alignment_score"),
                ]:
                    x, y = data(
                        cfg["eval_examples"],
                        seed_for(seed, generation, 0, split),
                        env,
                        split,
                    )
                    for name, column in [("robust", 0), ("shortcut", 1), ("cheap", 2)]:
                        close(
                            row["feature_policies"][name][metric],
                            float(((x[:, column] > 0) == y.bool()).float().mean()),
                        )
                        feature_scores += 1
                records_checked += len(group)
        # Independently train and score the first generation after each saved boundary.
        for arm in arms:
            path = folder / arm
            payload = read(path / "trajectory.json.gz")
            controls = arms[arm]["controls"]
            for filename, index in [
                (folder / "pre_challenge.pt", 0),
                (path / "pre_recovery.pt", challenge),
            ]:
                state = torch.load(filename, map_location="cpu", weights_only=True)
                row = payload["history"][index]
                generation = row["generation"]
                env = row["environment"]
                for slot, agent in enumerate(state["agents"]):
                    weights = {
                        name: value.clone()
                        .detach()
                        .requires_grad_(
                            not (
                                controls["freeze"] == "all"
                                or (
                                    controls["freeze"] == "robust"
                                    and name.startswith("robust_head.")
                                )
                            )
                        )
                        for name, value in agent["weights"].items()
                    }
                    if controls["freeze"] != "all":
                        opt = torch.optim.Adam(
                            list(weights.values()), lr=cfg["learning_rate"]
                        )
                        x, y = data(
                            cfg["train_examples"],
                            seed_for(seed, generation, slot, "train"),
                            env,
                            "train",
                        )
                        for _ in range(cfg["life_epochs"]):
                            opt.zero_grad(set_to_none=True)
                            out, r, s, c = forward(weights, agent["traits"], x)
                            loss = F.binary_cross_entropy_with_logits(out, y)
                            for branch, coefficient in [
                                (r, cfg["branch_aux_weight_robust"]),
                                (s, cfg["branch_aux_weight_shortcut"]),
                                (c, cfg["branch_aux_weight_cheap"]),
                            ]:
                                loss = (
                                    loss
                                    + coefficient
                                    * F.binary_cross_entropy_with_logits(branch, y)
                                )
                            loss.backward()
                            opt.step()
                    record = payload["individuals"][
                        index * cfg["population_size"] + slot
                    ]
                    with torch.no_grad():
                        for split, metric in [
                            ("fitness", "fitness_capability"),
                            ("report", "capability"),
                            ("conflict", "alignment_score"),
                        ]:
                            x, y = data(
                                cfg["eval_examples"],
                                seed_for(seed, generation, 0, split),
                                env,
                                split,
                            )
                            logits = forward(weights, agent["traits"], x)[0]
                            close(
                                float(((logits > 0) == y.bool()).float().mean()),
                                record[metric],
                            )
                            prediction_comparisons += len(y)
                    boundary_agents += 1
    # Independently assemble paired finite differences, without the production sign formula.
    contrast_count = 0

    def cell(bits, seed, phase, metric):
        name = f"b{bits[0]}_s{bits[1]}_c{bits[2]}"
        return cells[(name, seed)][challenge - 1 if phase == "end_challenge" else -1][
            metric
        ]

    for size in (1, 2, 3):
        for axes in itertools.combinations(range(3), size):
            name = "_x_".join(("bypass", "shift_removal", "cost")[i] for i in axes)
            others = [i for i in range(3) if i not in axes]
            for phase in ("end_challenge", "end_recovery"):
                for metric in METRICS:
                    diffs = []
                    for seed in seeds:

                        def difference(bits, remaining):
                            if not remaining:
                                return cell(bits, seed, phase, metric)
                            axis = remaining[0]
                            lo = list(bits)
                            hi = list(bits)
                            lo[axis] = 0
                            hi[axis] = 1
                            return difference(hi, remaining[1:]) - difference(
                                lo, remaining[1:]
                            )

                        effects = []
                        for settings in itertools.product((0, 1), repeat=len(others)):
                            bits = [0, 0, 0]
                            for axis, value in zip(others, settings):
                                bits[axis] = value
                            effects.append(difference(bits, list(axes)))
                        diffs.append(statistics.mean(effects))
                    reported = summary["contrasts"][name][phase][metric]
                    for a, b in zip(diffs, reported["paired_values"]):
                        close(a, b)
                    close(reported["mean"], statistics.mean(diffs))
                    close(
                        reported["std"],
                        statistics.stdev(diffs) if len(diffs) > 1 else 0,
                    )
                    if len(diffs) > 1:
                        rng = np.random.default_rng(260511)
                        ids = rng.integers(len(diffs), size=(5000, len(diffs)))
                        means = [sum(diffs[i] for i in row) / len(diffs) for row in ids]
                        for a, b in zip(
                            np.quantile(means, [0.025, 0.975]),
                            reported["bootstrap_percentile_95"],
                        ):
                            close(a, b)
                    assert reported["n"] == len(seeds)
                    contrast_count += 1
    return {
        "status": "PASS",
        "root_seeds": len(seeds),
        "arm_root_combinations": len(arms) * len(seeds),
        "individual_records_checked": records_checked,
        "fitness_values_recomputed": fitness_checked,
        "feature_policy_scores_recomputed": feature_scores,
        "boundary_agents_independently_trained_and_scored": boundary_agents,
        "binary_predictions_scored": prediction_comparisons,
        "factorial_metric_contrasts_recomputed": contrast_count,
        "scope": "Independent direct-tensor model/training and arithmetic reference using the same PyTorch Adam dependency; scores checked against ledgers, not a separate external certification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
