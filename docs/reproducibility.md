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

The test suite checks dataset semantics and validation, stable trait transforms, clone isolation, parent-pool eligibility, deterministic same-seed evolution, separate data streams, nonfinite training failures, configuration loading, history integrity, and metric aggregation.

## Reference suite

```bash
bash scripts/run_reference_suite.sh
```

Equivalent command:

```bash
python -m experiments.run_all --config configs/reference.yaml --out results
```

The command reads `configs/reference.yaml`. It accepts every `EvolutionConfig` field, including the three branch auxiliary-loss weights, plus a `seeds` list. Unknown fields and invalid types or ranges are rejected. `--seeds`, `--generations`, and `--population` explicitly override the corresponding YAML values; omitted settings use the dataclass defaults. Without `--config`, the same defaults and five reference seeds are used.

Seeds must be distinct built-in integers in `[0, 2**32)`, excluding booleans. Both Python experiment runners and command-line entrypoints validate them before training. The shortcut experiment and full suite require at least 30 generations to include all three phases.

## Determinism controls

- Python, NumPy, and PyTorch are seeded.
- PyTorch deterministic algorithms are requested with warning-only handling for unsupported operations; this is not a guarantee of bitwise agreement across library versions or platforms.
- Training, fitness evaluation, and conflict-probe streams derive seeds from the root seed, generation, agent index, and stream identifier through NumPy `SeedSequence`. Separate coordinates replace overlapping arithmetic offsets. Finite pseudorandom seeds do not provide a collision-free guarantee for all possible configurations.
- Parent choice and trait mutation use a seeded Python random generator.
- The schedule within each regime is identical across root seeds. Different regimes deliberately use different environment parameters.
- Cloning preserves weights, traits, dtype, and training mode without drawing new random initialisation values.
- Same-seed histories are checked for exact repeatability within the tested runtime. CPU intra-operation execution uses one PyTorch thread.

## Outputs

The reference suite writes:

- `results/selection_regimes.json`
- `results/selection_regimes_summary.json`
- `results/shortcut_challenge.json`
- `results/shortcut_challenge_summary.json`
- `results/reference_summary.json`
- figures under `results/figures/`

The two full experiment files retain every generation of every seed, along with the effective configuration. They and `reference_summary.json` record Python and dependency versions, operating platform, CPU execution, actual PyTorch thread and determinism settings, and SHA-256 hashes of package and experiment Python source files. Source hashes identify the executed bytes without depending on a local Git checkout. Configuration values reflect any command-line overrides.

The summary files contain mean, sample standard deviation, and sample count across the five seed-level population summaries. Shaded figure bands use the same sample standard deviation. A single-seed run reports standard deviation zero as an output convention; it does not estimate between-run uncertainty. Aggregation rejects repeated seeds, different regimes or configurations, and incomplete or misordered generation histories. Nonfinite statistics and non-standard JSON numeric constants are rejected.

Use the recorded source hashes, effective configuration, and runtime versions when comparing regenerated results. Fixes to parent selection, cloning, and derived seeds change trajectories, so historical outputs produced by the earlier implementation should not be pooled with the corrected reference runs.

## Paired factorial and checkpoints

The historical source, configuration and committed reference outputs are unchanged. Run the separate paired protocol into a new directory:

```bash
python -m experiments.causal_factorial --config configs/causal.yaml --out results/local/causal-v1
```

The default run has five root seeds, ten shared initial generations, ten intervention generations, ten recovery generations, eight factorial conditions and four controls. `--prefix` and `--challenge` change the two phase durations; configured generations must leave a nonempty recovery phase. Each root saves its shared pre-challenge checkpoint once, then each arm saves a pre-recovery checkpoint, final checkpoint and compressed individual/generation ledger. `manifest.json` records effective settings and runtime provenance; `summary.json` reports root-level effects and continuation checks. Checkpoints use restricted `weights_only=True` loading.

Outputs are create-only. An interrupted run can use the same command with `--resume`; completed arms are reused, the unchanged arm is verified again, and changed simulation source or runtime is rejected. An incomplete file write is not silently replaced. Preserve that output directory and restart into a new one when a checkpoint or ledger is incomplete.

Validation on 10 September 2026 reproduced all 60 arm trajectories byte-for-byte in a second execution, with all 125 checkpoint states identical. Individual-level accounting was checked for 12,000 post-boundary records. The separate legacy rerun passed 158 tests, reproduced matched pre-challenge numerical histories exactly and verified a mean final probe deficit of 0.0493359375; the largest difference from committed historical numerical cells was 2.22e-16. The paired-protocol tests additionally exercise serialized continuation, reporting independence, random elites, frozen traits/weights, mutation accounting, interrupted-run reuse and invalid evidence rejection.
