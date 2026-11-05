# Independent release verification

## Scope and original evidence

Version 0.2.0, reviewed on 5 November 2026. Original candidate
f733f1965a928d13ef3828b6bf25c01b4838348e; original base
3c90ec22349ed30abde3933b2bdd7cefb085bec6. A fresh source checkout and a new
Python 3.13 environment passed the candidate's 277 tests.

A separately authored 30-case review file reproduced 23 failures and seven passing
controls before repair. The original external probe remains unchanged in the local
review evidence. The release includes an expanded 45-case suite, including valid
mixed-mode/double-precision lineages, declared reporting sample counts, freezing,
malformed metadata and interrupted-run controls. All 45 pass after repair.

The 28 original implementation probes were independently replayed at the untouched
base and reproduced all 28 failures. They all pass at the repaired source.
These are independently designed software checks, not external human certification.

## Findings and repairs

### Typed study and resume contracts

The checker accepted additional or contradictory prefix/arm fields, a boolean seed
alias in prefix evidence, malformed execution provenance and inconsistent records
on the interrupted-run path. Shared validators now enforce the exact record fields,
canonical identity, configuration and explicit controls before reusing evidence.
Resume requires the full recorded source set, package versions and Torch settings
to match. A malformed resumed arm cannot produce a completed study.

### Statistical and feature-control evidence

A fabricated proportion could pass even when no integer number of correct answers
out of the declared sample size could produce it. Proportions must now correspond
to a possible float32 mean for that denominator. Separately reported feature-control
scores are regenerated from the declared report/conflict data streams rather than
accepted merely because they lie in [0, 1]. Per-generation fields are complete and
fixed; unspecified sample-count claims cannot be attached to accepted evidence.

### Boundary-state and inheritance consistency

Segment validation now checks both checkpoint payloads, fixed configuration, ending
controls, inherited parameter precision, per-module modes and trainability. Parameters
that cannot have learned under the declared controls are reconstructed through the
recorded inheritance and weight mutations. This detects altered frozen or no-learning
weights. It does not generally reconstruct arbitrary learned weights; the separate
neural reference below addresses selected recorded boundary generations.

All corrections preserve the simulator's valid numerical behavior and the original
research files. Schema-1 checkpoints retain their documented uniform-mode assumption.
The 1e-12 float ancestry tolerance remains limited to arithmetic roundoff; IDs,
types, integer counts, generator states, hashes and fingerprints remain exact.

## Independent numerical reference

`scripts/independent_reference.py` reconstructs expected results with direct tensor
operations. It does not call SelectionAgent.forward, _train_agent, advance,
validate_segment or the production summary functions to construct expectations.
The reference independently generates the synthetic data streams, runs the declared
linear-branch model and loss, and uses the same PyTorch Adam dependency.

For the five-root, fourteen-arm checked study it verifies:

- 14,500 individual fitness values and recorded selection/inheritance accounting.
- 8,700 feature-policy accuracy values from the regenerated observations.
- 1,400 boundary-agent training/inference cases, covering the first generation after
  each pre-challenge and pre-recovery checkpoint. Their recorded fitness, report and
  conflict-probe scores match, over 2,150,400 binary classifications in total.
- All 56 factorial metric/phase contrasts using recursive finite differences rather
  than the implementation's sign-product sum, including paired root-level summaries
  and seeded percentile intervals.

The classification count describes the scores recomputed, not millions of independent
experimental units. The five root seeds remain the sample size. Shared primitive
libraries and Adam mean this is not an independently implemented numerical stack.

## Full reproduction and historical preservation

An independent execution at the untouched candidate reproduced all 223 study files
byte-for-byte. The release source is rerun separately and compared to the checked
study. Only source/manifest/summary provenance fields are expected to change; all
220 raw checkpoints and ledgers must remain byte-identical. No checked study file
is rewritten to make it match the new version. Source_matches_current can therefore
be false when a historical study is verified by the newer code.

The nine pre-existing checked result/configuration files and all 223 implementation
study files remain unchanged. Negative outcomes, variation between roots, recovery
results and interactions remain inspectable. No tuning was performed using the
independent reruns, and replays do not add independent samples.

## Verification and release gates

The complete local suite contains 322 passing tests. The original external review
file passes 30/30, the expanded release suite passes 45/45 and the original audit
passes 28/28. The eight-case showcase remains supported. Clean non-editable wheel
verification checks every runtime and experiment module, runs the full tagged suite,
the independent reference, the archived study, the showcase and a newly executed
small factorial outside the source checkout.

Before publication, final-head CI must pass all eight jobs covering Linux Python
3.10-3.14, macOS 15, the reference dependency audit and installed-package verification.
The release workflow repeats exact-commit CI, verifies package digests and reviewed
main ancestry, refuses to replace existing tags, publishes the exact checked files
and tests their public downloads. The final workflow record is authoritative for
publication status; this document records the review and gates, not a promise that
an in-progress workflow has succeeded.

## Limits

This is synthetic classifier selection research, not a model of human influence or
proof that alignment persists in advanced agents. The new shift split is conditional
on baseline bypass reliability and verifier cost. Bootstrap intervals are descriptive
with five roots, not multiplicity-adjusted significance tests. File hashes establish
consistency, not producer authenticity. Restricted checkpoint loading is not a
sandbox for arbitrary external files. Cross-platform bitwise training identity is
not guaranteed by the supported dependency ranges.

Primary implementation guidance consulted:

- https://docs.pytorch.org/docs/stable/notes/serialization.html
- https://docs.pytorch.org/docs/stable/notes/randomness.html
