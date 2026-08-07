# Method

## Research question

This benchmark asks whether an alignment-linked verification trait is selected for, selected against, or left to drift when its contribution to capability changes.

The model is deliberately small so the selection mechanism can be inspected directly and rerun on CPU.

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

Alignment-probe performance is recorded but is not directly included in fitness.

The top-performing agents form the parent pool. Two elites are retained and the remaining population is produced by cloning and mutating selected parents.

`selection_differential` records the mean verifier strength among selected parents minus the population mean before reproduction.

## Regimes

### Safety-only

The ordinary shortcut remains reliable and verification has an explicit cost. The alignment-linked trait has safety value on the held-out conflict probe but no capability benefit in the selection environment.

### Neutral

The shortcut remains reliable and verification has no explicit cost. The trait is approximately fitness-neutral and can drift through mutation and hitchhiking.

### Capability-positive

Training and evaluation include distribution-shift episodes where the ordinary shortcut reverses. Verification now improves task capability as well as the alignment probe, while retaining a small cost.

## Alignment probe

The held-out alignment probe forces both the ordinary shortcut and cheap bypass features to point toward the wrong label while the robust feature remains informative.

Accuracy on this conflict set measures whether the agent's verification path can resist the misleading shortcut.

This is a controlled proxy for an alignment-relevant property, not a model of human values or full corrigibility.

## Cheap-route challenge

The shortcut challenge has three phases:

1. generations 0 to 9 use the capability-positive environment,
2. generations 10 to 19 introduce a highly predictive cheap bypass feature and remove the shifted episodes that made verification capability-useful,
3. generations 20 to 29 restore the capability-positive environment and remove the cheap shortcut advantage.

The challenge asks whether a verifier that had been positively selected remains stable when the fitness landscape temporarily favors an easier route around it, and whether selection can recover the verifier after the environment changes again.

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

Reference results use five fixed seeds: `7`, `17`, `29`, `41`, and `53`.
