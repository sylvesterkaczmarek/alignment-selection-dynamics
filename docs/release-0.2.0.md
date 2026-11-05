# Alignment Selection Dynamics 0.2.0

This release completes the paired-evidence hardening and independent review in PR #4.
It includes the existing factorial design and the new five-root, fourteen-arm study
separating removal of training shift from removal of selection-evaluation shift.

## Corrections

Typed study/resume contracts reject contradictory fields, configuration aliases,
malformed provenance and incompatible reused checkpoints. Sample-denominator checks
and regenerated feature policies strengthen evidence validation. Boundary states
preserve declared controls, trainability, precision, inherited modes and known
frozen/no-learning weights. Original valid numerical results remain unchanged.

## Verification

The source suite has 322 passing tests, including 45 release-review cases. An
independent direct-tensor reference recomputes 14,500 fitness/selection records,
8,700 feature scores, 1,400 boundary-agent training/inference cases and 56 factorial
metric contrasts. The unchanged candidate study reproduced all 223 files exactly.
All original histories, uncertainty and negative findings remain inspectable.

The eight-case demonstration is available with:

```sh
python -m experiments.showcase --out /tmp/asd-showcase-new
```

Use a new output directory. Install the wheel attached to this GitHub release using
`python -m pip install /path/to/alignment_selection_dynamics-0.2.0-py3-none-any.whl`.
This is a GitHub-only release; no PyPI upload was performed.

## Scope

Five roots remain the independent sample. The study is a small synthetic classifier
benchmark with imposed routing traits and verifier cost. It establishes neither
human-influence effects nor general alignment preservation. Checksums establish
internal consistency, not source authenticity. The independent review file and
source archive are attached alongside the exact tested distribution artifacts.
