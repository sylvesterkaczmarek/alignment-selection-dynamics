"""Recompute selection and inheritance accounting from complete per-agent ledgers."""

from __future__ import annotations
from dataclasses import asdict
import math
import random
import numpy as np
import torch
from .causal import checkpoint_fingerprint, protocol_seed
from .integrity import canonical, finite, validate_tree
from .model import logistic

SCORES = (
    "fitness_capability",
    "fitness",
    "verification_cost",
    "verifier_strength",
    "bypass_strength",
    "verifier_logit",
    "bypass_logit",
    "capability",
    "alignment_score",
)
RATES = {
    "fitness_capability",
    "verifier_strength",
    "bypass_strength",
    "capability",
    "alignment_score",
}


def close(actual, expected, name):
    finite(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"inconsistent {name}")


def same_ancestry(left, right):
    """Exact structure/types/identity, rounding tolerance only for float arithmetic.

    Python random.gauss uses platform libm: the same random draws can differ by
    one float64 ULP on macOS versus Linux. IDs, integer counts and RNG bits never
    use this tolerance. Raw file hashes and checkpoint fingerprints stay exact.
    """
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            same_ancestry(left[k], right[k]) for k in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            same_ancestry(a, b) for a, b in zip(left, right)
        )
    return left == right


def validate_segment(history, individuals, start, end, cfg, controls, environments):
    """Checks arithmetic and recorded random draws, without rerunning training."""
    validate_tree(history)
    validate_tree(individuals)
    if (
        len(history) != len(environments)
        or len(individuals) != len(history) * cfg.population_size
    ):
        raise ValueError("incomplete segment history or individual ledger")
    if end.seed != start.seed or end.next_generation != start.next_generation + len(
        history
    ):
        raise ValueError("checkpoint segment identity mismatch")
    parent_rng = random.Random()
    parent_rng.setstate(start.parent_state)
    trait_rng = random.Random()
    trait_rng.setstate(start.trait_state)
    weight_rng = torch.Generator().set_state(start.weight_state)
    births = start.births
    traits = [(a.verifier_logit, a.bypass_logit) for a in start.population]
    dtypes = [next(a.parameters()).dtype for a in start.population]
    shapes = {
        name: tuple(p.shape) for name, p in start.population[0].named_parameters()
    }
    for index, (row, env) in enumerate(zip(history, environments)):
        generation = start.next_generation + index
        if (
            type(row["generation"]) is not int
            or row["generation"] != generation
            or canonical(row["environment"]) != canonical(asdict(env))
        ):
            raise ValueError("generation or environment differs from declared segment")
        if canonical(row["data_seeds"]) != canonical(
            {
                n: protocol_seed(start.seed, generation, 0, n)
                for n in ("fitness", "report", "conflict")
            }
        ):
            raise ValueError("invalid fitness/report/probe stream binding")
        group = individuals[
            index * cfg.population_size : (index + 1) * cfg.population_size
        ]
        for slot, record in enumerate(group):
            if record.keys() != set(SCORES) | {
                "id",
                "parent_id",
                "mutation",
                "eligible",
                "elite_copies",
                "offspring_count",
                "generation",
                "train_seed",
            }:
                raise ValueError("unexpected or missing individual fields")
            if (
                type(record["generation"]) is not int
                or record["generation"] != generation
                or not same_ancestry(
                    {k: record[k] for k in ("id", "parent_id", "mutation")},
                    births[slot],
                )
                or type(record["train_seed"]) is not int
                or record["train_seed"]
                != protocol_seed(start.seed, generation, slot, "train")
            ):
                raise ValueError("individual ancestry or training stream mismatch")
            for key in SCORES:
                finite(record[key], key, probability=key in RATES)
            for key, expected in zip(("verifier_logit", "bypass_logit"), traits[slot]):
                close(record[key], expected, key)
            close(
                record["verifier_strength"],
                logistic(record["verifier_logit"]),
                "verifier strength",
            )
            close(
                record["bypass_strength"],
                logistic(record["bypass_logit"]),
                "bypass strength",
            )
            close(
                record["verification_cost"],
                env.verifier_cost * record["verifier_strength"],
                "verification cost",
            )
            close(
                record["fitness"],
                record["fitness_capability"] - record["verification_cost"],
                "fitness",
            )
            if (
                type(record["eligible"]) is not bool
                or type(record["elite_copies"]) is not int
                or type(record["offspring_count"]) is not int
            ):
                raise ValueError(
                    "selection decisions and counts must have canonical types"
                )
        for key in SCORES:
            close(
                row[key],
                float(np.mean([record[key] for record in group])),
                key + " population mean",
            )
        close(row["best_fitness"], max(v["fitness"] for v in group), "best fitness")
        feature = row["feature_policies"]
        if set(feature) != {"robust", "shortcut", "cheap"}:
            raise ValueError("incomplete feature controls")
        for values in feature.values():
            if set(values) != {"capability", "alignment_score"}:
                raise ValueError("incomplete feature metrics")
            for value in values.values():
                finite(value, "feature accuracy", probability=True)
        order = (
            parent_rng.sample(range(cfg.population_size), cfg.population_size)
            if controls.selection == "random"
            else sorted(range(cfg.population_size), key=lambda i: -group[i]["fitness"])
        )
        eligible = order[: cfg.parent_pool]
        chosen = eligible[: cfg.elites] + [
            parent_rng.choice(eligible) for _ in range(cfg.population_size - cfg.elites)
        ]
        for slot, record in enumerate(group):
            if (
                record["eligible"] is not (slot in eligible)
                or record["elite_copies"] != int(slot in eligible[: cfg.elites])
                or record["offspring_count"] != chosen.count(slot)
            ):
                raise ValueError(
                    "recorded selection differs from fitness/random policy"
                )
        next_births = []
        next_traits = []
        for slot, parent in enumerate(chosen):
            mutation = {"verifier_logit": 0.0, "bypass_logit": 0.0, "weights": {}}
            if slot >= cfg.elites:
                for name in ("verifier_logit", "bypass_logit"):
                    delta = trait_rng.gauss(0.0, cfg.trait_mutation_std)
                    if controls.mutate_traits:
                        mutation[name] = delta
                for name, shape in shapes.items():
                    delta = cfg.weight_mutation_std * torch.randn(
                        shape, dtype=dtypes[parent], generator=weight_rng
                    )
                    if controls.freeze == "all" or (
                        controls.freeze == "robust" and name.startswith("robust_head.")
                    ):
                        delta.zero_()
                    mutation["weights"][name] = delta.tolist()
            next_traits.append(
                (
                    traits[parent][0] + mutation["verifier_logit"],
                    traits[parent][1] + mutation["bypass_logit"],
                )
            )
            next_births.append(
                {
                    "id": f"{start.seed}:{generation + 1}:{slot}",
                    "parent_id": births[parent]["id"],
                    "mutation": mutation,
                }
            )
        old_mean = float(np.mean([logistic(v[0]) for v in traits]))
        parent_mean = float(np.mean([logistic(traits[i][0]) for i in chosen]))
        pool_mean = float(np.mean([logistic(traits[i][0]) for i in eligible]))
        newborn = float(np.mean([logistic(v[0]) for v in next_traits]))
        for key, expected in [
            ("selection_differential", pool_mean - old_mean),
            ("realized_selection_differential", parent_mean - old_mean),
            ("mutation_change", newborn - parent_mean),
            ("newborn_verifier_change", newborn - old_mean),
        ]:
            close(row[key], expected, key)
        births, traits, dtypes = next_births, next_traits, [dtypes[p] for p in chosen]
    if not same_ancestry(end.births, births):
        raise ValueError("final ancestry does not match reproduced selections")
    for agent, pair in zip(end.population, traits):
        close(agent.verifier_logit, pair[0], "final verifier logit")
        close(agent.bypass_logit, pair[1], "final bypass logit")
    if (
        end.parent_state != parent_rng.getstate()
        or end.trait_state != trait_rng.getstate()
        or not torch.equal(end.weight_state, weight_rng.get_state())
    ):
        raise ValueError("final generator state does not match recorded evolution")
    return {
        "generations": len(history),
        "individual_records": len(individuals),
        "final_fingerprint": checkpoint_fingerprint(end),
    }
