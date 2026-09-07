# Limitations

This repository is a controlled selection-dynamics benchmark. It should not be read as evidence that evolutionary pressure will preserve alignment in advanced systems.

- The agents are small neural classifiers rather than language models or general agents.
- The inherited verifier and bypass traits are explicit scalar gates. Real alignment-relevant mechanisms would be distributed across representations, circuits, objectives, scaffolds, and external systems.
- The alignment probe measures resistance to a synthetic misleading shortcut. It is not a complete model of corrigibility, honesty, value learning, or human intent.
- Generational selection is a stylized analogue of repeated capability improvement. It is not recursive self-improvement.
- The fitness function is explicit and stationary within each phase. Real deployment incentives would be more complex and only partially observed.
- The regime names are historical identifiers. In particular, `neutral` removes the explicit verifier penalty but does not remove the verifier's effect on accuracy or fitness. The `safety_only` regime likewise does not guarantee that verification has no capability benefit.
- The cheap-route challenge jointly removes distribution shift, makes the bypass feature predictive, and increases the verifier penalty from `0.01` to `0.03`. Its outcomes cannot isolate the causal contribution of any one change, or show that advanced systems will discover this particular bypass.
- The verification cost is an imposed fitness penalty. All model branches are evaluated; the benchmark does not establish computational or energy savings for a cheap route.
- Each agent receives a different finite evaluation sample. Sampling noise can affect parent rankings, even for equally capable policies. Stable tie-breaking also preserves population order rather than randomising equal-fitness agents.
- Five distinct seeds provide a reproducibility snapshot rather than strong inferential evidence. Sample standard deviations describe between-run variation; they are not confidence intervals. Reusing a seed does not create an additional independent run and is rejected by the experiment runners.
- The selection differential describes the eligible parent pool without weighting by realised reproductive success. Elitism, random parent sampling, mutation, and subsequent learning also affect trait trajectories.
- Coordinate-derived pseudorandom streams separate training, fitness evaluation, and conflict probes. They remove the previous overlapping arithmetic seed offsets but are not a mathematical guarantee against every possible seed collision or statistical dependence.
- The model architecture and environment parameters were chosen to make the hypothesized selection regimes experimentally observable in a small CPU-sized benchmark.

The intended use is mechanistic hypothesis testing: make the selection pressures explicit, measure what changes, and identify failure modes that larger experiments should investigate.
