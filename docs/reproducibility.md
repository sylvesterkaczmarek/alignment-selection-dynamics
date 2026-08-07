# Reproducibility

## Reference environment

The project requires Python 3.10 or newer and uses PyTorch, NumPy, Matplotlib, PyYAML, and pytest.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

The test suite checks dataset semantics, trait bounds, cloning, deterministic same-seed evolution, history fields, and metric aggregation.

## Reference suite

```bash
bash scripts/run_reference_suite.sh
```

Equivalent command:

```bash
python -m experiments.run_all --seeds 7 17 29 41 53 --out results
```

The reference configuration is also recorded in `configs/reference.yaml`.

## Determinism controls

- Python, NumPy, and PyTorch are seeded.
- PyTorch deterministic algorithms are requested where available.
- Each agent-generation pair receives a deterministic derived data seed.
- Parent choice and trait mutation use a seeded Python random generator.
- Identical environment schedules are used across seeds and compared regimes.
- The same seed produces the same serialized history in the regression test.

## Outputs

The reference suite writes:

- `results/selection_regimes.json`
- `results/selection_regimes_summary.json`
- `results/shortcut_challenge.json`
- `results/shortcut_challenge_summary.json`
- `results/reference_summary.json`
- figures under `results/figures/`

The summary files contain mean, sample standard deviation, and sample count for the five reference seeds.
