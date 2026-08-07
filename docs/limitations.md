# Limitations

This repository is a controlled selection-dynamics benchmark. It should not be read as evidence that evolutionary pressure will preserve alignment in advanced systems.

- The agents are small neural classifiers rather than language models or general agents.
- The inherited verifier and bypass traits are explicit scalar gates. Real alignment-relevant mechanisms would be distributed across representations, circuits, objectives, scaffolds, and external systems.
- The alignment probe measures resistance to a synthetic misleading shortcut. It is not a complete model of corrigibility, honesty, value learning, or human intent.
- Generational selection is a stylized analogue of repeated capability improvement. It is not recursive self-improvement.
- The fitness function is explicit and stationary within each phase. Real deployment incentives would be more complex and only partially observed.
- The cheap-route challenge is deliberately constructed. It demonstrates that positive selection can reverse when the capability landscape changes, not that advanced systems will discover this exact bypass.
- Five seeds provide a reproducibility snapshot rather than strong inferential evidence.
- The model architecture and environment parameters were chosen to make the hypothesized selection regimes experimentally observable in a small CPU-sized benchmark.

The intended use is mechanistic hypothesis testing: make the selection pressures explicit, measure what changes, and identify failure modes that larger experiments should investigate.
