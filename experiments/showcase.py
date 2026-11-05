"""Eight small CPU checks illustrating the paired experiment's contracts."""

from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from alignment_selection_dynamics.causal import (
    Controls,
    initial_checkpoint,
    advance,
    save_checkpoint,
    load_checkpoint,
    checkpoint_fingerprint,
)
from alignment_selection_dynamics.evolution import EvolutionConfig
from alignment_selection_dynamics.io import write_json
from alignment_selection_dynamics.metrics import summarize
from .causal_factorial import BASELINE, study_arms, summarize_factorial


def demonstrate(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = EvolutionConfig(
        population_size=3,
        parent_pool=2,
        elites=1,
        generations=3,
        life_epochs=1,
        train_examples=8,
        eval_examples=16,
    )
    initial = initial_checkpoint(73, cfg)
    original = checkpoint_fingerprint(initial)
    full, history, records = advance(initial, [BASELINE] * 3, cfg)
    prefix, first, first_records = advance(initial, [BASELINE], cfg)
    save_checkpoint(out / "prefix.pt", prefix)
    reloaded = load_checkpoint(out / "prefix.pt")
    end, rest, rest_records = advance(reloaded, [BASELINE] * 2, cfg)
    results = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        results.append({"case": name, "status": "PASS"})
        print(name, "PASS", flush=True)

    check("input_checkpoint_is_preserved", checkpoint_fingerprint(initial) == original)
    check(
        "checkpoint_continuation_matches",
        checkpoint_fingerprint(full) == checkpoint_fingerprint(end)
        and history == first + rest
        and records == first_records + rest_records,
    )
    no_report, _, _ = advance(initial, [BASELINE] * 3, cfg, diagnostics=False)
    check(
        "reporting_does_not_change_selection",
        checkpoint_fingerprint(full) == checkpoint_fingerprint(no_report),
    )
    check(
        "offspring_accounting_is_complete",
        all(
            sum(r["offspring_count"] for r in records if r["generation"] == g)
            == cfg.population_size
            for g in range(3)
        ),
    )
    fixed, _, _ = advance(initial, [BASELINE], cfg, Controls(mutate_traits=False))
    check(
        "fixed_trait_control_has_no_trait_mutation",
        all(
            b["mutation"]["verifier_logit"] == b["mutation"]["bypass_logit"] == 0
            for b in fixed.births
        ),
    )
    arms = study_arms(True)
    check(
        "training_and_evaluation_shifts_are_separate",
        arms["train_shift_only"][0].eval_shift_fraction == BASELINE.eval_shift_fraction
        and arms["eval_shift_only"][0].train_shift_fraction
        == BASELINE.train_shift_fraction,
    )
    rejected = False
    try:
        summarize([True, False])
    except ValueError:
        rejected = True
    check("boolean_metrics_are_rejected", rejected)
    cohorts = {
        name: [
            {
                "seed": 7,
                "history": [
                    {
                        "generation": g,
                        **{
                            k: 0.5
                            for k in (
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
        for name in arms
        if name.startswith("b") and not name.endswith(("random", "traits"))
    }
    cohorts["b1_s1_c1"].append(deepcopy(cohorts["b1_s1_c1"][0]))
    rejected = False
    try:
        summarize_factorial(cohorts, 1)
    except ValueError:
        rejected = True
    check("duplicate_experiment_roots_are_rejected", rejected)
    result = {
        "status": "PASS",
        "cases": results,
        "passed": len(results),
        "scope": "Synthetic CPU classifiers and explicit evidence contracts. No claim about human influence or general model safety.",
    }
    write_json(out / "summary.json", result, exclusive=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    print(json.dumps(demonstrate(parser.parse_args().out), indent=2))
