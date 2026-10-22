# Project instructions

Current authority: MASTER_SPEC.md (MASTER_SPEC_v1.0, adopted 20 October 2026).
Read docs/SPEC_ALIGNMENT.md for verified reuse, requirement changes and evidence gaps.
The original docs/MASTER_BUILD_PROMPT.md and v1 reports are historical records.

Current deployed action semantics are AUTO_NONREFERABLE, RECAPTURE and
SPECIALIST_REVIEW. Implement these in the versioned qsart/triage_v2.py; preserve
qsart/triage.py and all frozen v1 results. Never rename positive AUTO_SCREEN
outputs into non-referable outcomes. Report overload when mandatory deferrals
exceed independent integer capacities. Probability/uncertainty gates are research
rules, not validated clinical safety guarantees.

Preserve the consumed 340-image confirmation population as historical evaluation.
Future fitting, threshold tuning and model selection use development data only.
Official test and IDRiD external evaluation remain reserved. APTOS access must
follow its official source/terms; never substitute an unverified image mirror.
Conformal calibration must be separate from temperature fitting and model
selection; account for patient groups and state the exchangeability limitations.
Study notes distinguish paper-inspired methods from exact reproduction.

Follow the master specification's parallel workstreams only with explicit file
ownership to avoid agents overwriting each other. Keep one authoritative backlog.
Preserve formal manuscript drafting until substantive experiments are complete;
retain the existing draft as a historical working document.

QSART is a medical imaging research prototype. Never invent model predictions,
training results, references, timings, dataset fields, or clinical validation.
Read PROJECT_AUDIT.md, TASKS.md and PROGRESS.md before extending the project.
Preserve original assets and working starter code. Keep patient images and
identifiers, datasets, local paths, checkpoints and credentials out of Git.

Use real authorized images for scientific evidence and explicitly labeled
synthetic fixtures for software tests. Reject missing/untrained/test model
artifacts at production readiness. Do not tune on held-out test labels. Keep
all known patient views together. Report unknown grouping limitations.

After changes run relevant pytest tests and Ruff. Record actual commands,
results and remaining blockers in PROGRESS.md. Report deployment as unverified
until a real container is exercised with a genuinely trained checkpoint.
