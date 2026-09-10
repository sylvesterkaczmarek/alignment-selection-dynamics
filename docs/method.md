# Method

## Research question

This benchmark asks whether an alignment-linked verification trait is selected for, selected against, or left to drift when its contribution to capability changes.

The model is deliberately small so the selection mechanism can be inspected directly and rerun on CPU. The regime names describe intended interventions; a trait's actual contribution to fitness must be measured.

## Agent

Each agent contains three learned branches:

- a robust branch trained on a noisy but stable signal,
- a shortcut branch trained on a cheap signal that is highly predictive in ordinary conditions but can reverse under shift,
- a cheap bypass branch that can become useful during the route-around challenge.

Two inherited scalar traits control how these branches affect behavior:

- `verifier_strength` controls how strongly disagreement between the shortcut and robust branches is corrected toward robust evidence,
- `bypass_strength` controls how strongly a separate cheap path overrides the verified path.

The verifier and bypass traits are represented as logits and mapped to `[0, 1]` with a logistic transform.

## Within-generation learning

Each generation inherits both neural weights and the two scalar traits from selected parents. Descendants receive small mutations to weights and traits, then train on the current environment.

The training objective includes the final policy loss plus auxiliary losses on the individual branches. This keeps the experiment focused on selection over the inherited routing traits rather than allowing a branch to become useless solely because its gate was temporarily small.

## Selection

Fitness is:

```text
fitness = capability_accuracy - verifier_cost * verifier_strength
```

Alignment-probe performance is recorded but is not directly included in fitness. `verifier_cost` is an imposed fitness-penalty coefficient. The benchmark does not measure verification latency, energy consumption, or computational savings from bypassing it.

The top `parent_pool` agents form the eligible parent pool. Fitness ties preserve the population's existing order. The top `elites` agents are retained without mutation; each other descendant draws a parent uniformly from the full eligible pool and mutates its traits and weights. The reference configuration uses five eligible parents and two elites in a population of ten.

`selection_differential` records the unweighted mean verifier strength across the full eligible parent pool minus the population mean before reproduction. It is not weighted by the realised number of descendants, including elites, and does not equal the generation-to-generation change after mutation and learning.

Each agent is evaluated on its own generated fitness sample. Different evaluation samples add noise to fitness rankings and parent selection. These rankings do not constitute a comparison on a shared fixed test set.

## Regimes

### Safety-only

The ordinary shortcut remains reliable and verification has an explicit penalty coefficient of `0.08`. This is intended to discourage verification when shift is absent. Verification can still change ordinary classification accuracy, so the historical `safety_only` identifier does not establish that the trait has zero capability benefit.

### Zero-cost control

The `neutral` identifier is retained for compatibility. The shortcut remains reliable and the explicit verifier penalty is zero. Changing verifier strength can nevertheless change classification accuracy and therefore fitness. This is a zero-cost control, not an established fitness-independent drift condition; selection and lineage effects can still act on the trait.

### Capability-positive

Training and evaluation include distribution-shift episodes where the ordinary shortcut reverses, with fractions `0.25` and `0.35` respectively. This creates an opportunity for verification to improve task capability and the alignment probe, with an imposed penalty coefficient of `0.01`. The resulting behaviour depends on the learned branches and routing traits.

## Alignment probe

The held-out alignment probe forces both the ordinary shortcut and cheap bypass features to point toward the wrong label while the robust feature remains informative.

Accuracy on this conflict set measures whether the agent's verification path can resist the misleading shortcut.

This is a controlled proxy for an alignment-relevant property, not a model of human values or full corrigibility.

## Cheap-route challenge

The shortcut challenge changes three aspects of the selection environment together:

| Phase | Generations | Training/evaluation shift | Training/evaluation bypass accuracy | Verifier penalty coefficient |
| --- | --- | --- | --- | --- |
| Before challenge | 0–9 | 0.25 / 0.35 | 0.5 / 0.5 | 0.01 |
| Cheap shortcut | 10–19 | 0 / 0 | 0.995 / 0.995 | 0.03 |
| Recovery | 20 onwards | 0.25 / 0.35 | 0.5 / 0.5 | 0.01 |

The default run ends at generation 29. Longer runs extend the recovery phase. Summary snapshots use generations 9, 19, and the final configured generation. The experiment runner requires at least 30 generations before it will start.

The challenge measures the response to this combined intervention. A decline in verification cannot be attributed specifically to the bypass route: removing shift and tripling the explicit verifier penalty also change selection pressure. Separating those causes would require additional controlled comparisons.

## Reported metrics

Each generation records:

- capability accuracy,
- alignment-probe accuracy,
- population mean verifier strength,
- population mean bypass strength,
- verifier cost,
- fitness,
- best fitness,
- verifier selection differential.

Generation metrics describe the evaluated population before reproduction. Reference results use five distinct fixed seeds: `7`, `17`, `29`, `41`, and `53`. Tables and shaded figure bands report mean and sample standard deviation across seed-level population means. The reported sample count is the number of runs, not the number of agents or generations. These are descriptive summaries, not confidence intervals or significance tests.

## Paired causal protocol

`experiments.causal_factorial` implements `paired-v1` separately from the historical runner above. It branches all eight combinations of bypass accuracy (0.5/0.995), train/evaluation shift (0.25/0.35 versus 0/0), and verifier penalty (0.01/0.03) from one population per root seed. The boundary is after generation 9 reproduction and mutation, before generation 10 learning. Weights, traits, ordering, ancestry and private parent, trait-mutation and weight-mutation random states are preserved. Adam restarts each generation, so no optimizer state crosses this boundary.

Every competing agent sees the same fresh fitness examples within a generation. Independent reporting and conflict batches never affect selection. Data coordinates match across interventions; descendants with matching population slots are not necessarily the same lineage after selection diverges. Initialization and all subsequent streams are private. Removing or resizing diagnostics leaves training and reproduction unchanged.

The unchanged arm is checked against a separately uninterrupted run, including every individual record and final checkpoint. Additional unchanged and combined-challenge arms use random parent eligibility and random elites, with no fitness-based reproduction, or protect both inherited logits from mutation. Protected traits can still change in population frequency through selection over initially different lineages. Robust-feature, shortcut-feature and cheap-feature policies are evaluated on the same reporting examples.

Agent identifiers are scoped by arm and root seed. Records retain parent identifiers, fitness and reporting scores, eligibility, actual offspring counts, and realized trait and weight mutations. The realized offspring-weighted selection differential plus the mean mutation-induced phenotype change equals the newborn mean verifier change, up to floating-point rounding. Symmetric logit mutation can pull sigmoid-transformed traits toward 0.5; it is not generally phenotype-neutral.

Factorial main effects average high-minus-low differences over the other two factors. A two-factor interaction is a difference of differences averaged over the third factor; the three-factor interaction is the corresponding third difference. All contrasts are paired within root seed. Sample standard deviations describe variation across five roots and are not confidence intervals.

In the executed exploratory cohort, changing bypass accuracy alone reduced generation-19 mean conflict-probe accuracy from 0.87758 to 0.31711 while mean verifier strength remained 0.71017. Removing shift alone reduced probe accuracy to 0.49332. Increasing cost alone gave 0.86457. The combined challenge's final recovery deficit was 0.02148 relative to its matched unchanged arm. These revised-stream results must not be pooled with the historical 0.04934 deficit. Gate strength alone does not establish retained function or explain recovery.
