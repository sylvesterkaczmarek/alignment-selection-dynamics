# Factorial validation study

## Fixed design

This extends the existing paired-v1 runner. The original eight-condition factorial
and four random-selection/fixed-trait controls were already present before this
update. The extension adds two arms at baseline bypass reliability and baseline
verifier cost, varying training-shift removal and selection-evaluation-shift
removal separately. Combined with the unchanged and joint-shift-removal arms,
these form a paired two-by-two comparison.

Root seeds 101, 103, 107, 109 and 113 were fixed before the new results were run.
They are distinct from the historical roots 7, 17, 29, 41 and 53. Every root uses
10 agents, 10 shared initial generations, 10 intervention generations and 10
recovery generations. The same pre-intervention population and random-stream
states seed all 14 arms. Complete runs cover 70 arm/root combinations and 14,500
individual-generation records, including the five shared prefixes.

Fitness evaluation is shared across agents within each generation. Reporting and
conflict-probe samples have separate streams and never contribute to parent
selection. The reporting capability distribution follows the evaluation setting;
the conflict-probe distribution remains the same across arms. Training and
selection-evaluation shift remain jointly varied in the historical three-factor
comparison, but are separated in the added two-by-two study.

Bypass reliability changes from 0.5 to 0.995 in both training and evaluation;
verifier cost changes from 0.01 to 0.03. Distribution shift changes from 0.25 in
training and 0.35 in evaluation to zero. Every arm recovers to the baseline
**environment** for ten generations. Random-selection and fixed-trait controls
retain their respective intervention during recovery.

## Recorded outcomes

Conflict-probe accuracy at the end of the ten-generation challenge, mean plus/minus
sample standard deviation across five independent root-level population means:

| Intervention | Probe accuracy |
| --- | ---: |
| Unchanged environment | 0.879 ± 0.018 |
| Higher bypass reliability only | 0.289 ± 0.299 |
| Joint training and evaluation shift removal only | 0.522 ± 0.144 |
| Higher verifier cost only | 0.866 ± 0.029 |
| All three historical factors changed | 0.239 ± 0.265 |
| Training shift removed only | 0.870 ± 0.018 |
| Selection-evaluation shift removed only | 0.563 ± 0.217 |

Across all eight factorial conditions, the mean end-challenge effect on the probe
is -0.367 for higher bypass reliability, -0.243 for joint shift removal and -0.080
for higher cost. The complete paired root values, interaction terms, capability
scores and verifier/bypass traits remain in the machine-readable summary.

At baseline bypass reliability and cost, the added two-by-two decomposition gives
an average training-shift-removal effect of -0.025 and an evaluation-shift-removal
effect of -0.332. This supports a stronger selection-evaluation effect in these
five roots under this fixed setup. It is not an architecture-independent or
population-level claim. The five roots and inherited descendants are not treated
as thousands of independent observations.

After recovery, the unchanged arm's mean probe score is 0.892 and the combined
challenge arm's is 0.869. The finite recovery horizon cannot establish permanent
loss or guaranteed recovery. All poor-performing roots are retained.

## Statistics

Main effects are high-minus-low differences averaged over other factors. The
existing k-factor interaction convention is the k-fold difference averaged over
remaining factors. Consequently, two-factor and three-factor numbers are not the
same scaling as a conventional two-level regression effect or coefficient. Tests
check every term independently using signed synthetic designs.

The primary variability measure is sample standard deviation across roots.
Seeded 5,000-resample percentile bootstrap intervals are descriptive and are
resampled at the paired root level. They are based on only five roots, are not
multiplicity-adjusted, and are not labelled confirmatory significance tests.
The code and environmental choices were informed by the earlier exploratory
study. New roots alone do not turn this into a preregistered external validation.

## Evidence and replay

`results/factorial-validation-v2` contains a manifest written before the run,
complete prefix/arm trajectories, checkpoints at all branch boundaries, the
summary, and a completion manifest with file checksums. The verifier checks
recorded fitness arithmetic, ordered identity, every offspring and elite count,
actual parent draws, trait/weight mutation draws, boundary generator states,
checkpoint fingerprints, and all seed-level summaries. Platform float arithmetic
uses a documented 1e-12 ancestry tolerance; IDs, counts and file/checkpoint hashes
remain exact. Summary-only recomputation permits 1e-11 relative rounding tolerance.

It does not independently establish that reported predictions were produced by
training. Separate execution replay checks that claim. Hashes establish internal
consistency, not authenticity of an evidence producer. PyTorch restricted loading
is not a sandbox for arbitrary untrusted checkpoints.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m experiments.causal_factorial --config configs/factorial-validation.yaml --split-shift --out /tmp/asd-factorial-new
python -m experiments.verify_factorial /tmp/asd-factorial-new
```

The source and runtime fingerprints identify the executed code. Identical seeds
are not a guarantee of bitwise equality across PyTorch versions or platforms.
Replaying the same five roots does not increase the independent sample count.

Primary methodology references:

- NIST/SEMATECH main and interaction effects: https://www.itl.nist.gov/div898/handbook/pri/section6/pri615.htm
- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch serialization limitations: https://docs.pytorch.org/docs/stable/notes/serialization.html
