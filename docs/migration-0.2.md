# Version 0.2 migration

Valid simulator parameters and numerical results are unchanged. The paired checkpoint
configuration is fixed, and Controls remains the explicit intervention interface.
Malformed study/provenance dictionaries, unknown ledger fields, impossible observed
sample proportions and contradictory boundary states are rejected. Resume requires
an unchanged complete recorded execution environment and source set. Use a new
output directory to run a changed source version; preserve incomplete historical
outputs for diagnosis rather than relabelling them.

The internal validate_segment helper accepts report_examples for a segment whose
reporting batches were explicitly resized. Its default is cfg.eval_examples. The
production factorial uses that default. Frozen/no-learning parameters, inherited
modes/precision and final trainability are checked against their recorded lineage.

All published checkpoint fields and existing valid schema-1/schema-2 checkpoint
loading remain supported. Ordinary checkpoint loading and study consistency checks
do not constitute a sandbox or authentication service for third-party evidence.
