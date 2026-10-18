# Project instructions

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
