# QSART project audit — 2026-10-18

## Inspected evidence

The parent workspace contains two literature-review DOCX documents, a Review-1
PowerPoint, a panel spreadsheet, a paper-library CSV/JSON, and the 22,909-byte
`QSART_Research_Starter_v0_1.zip`. Originals are preserved. The starter was
extracted into this dedicated project directory; the supplied build requirements
are preserved in `docs/MASTER_BUILD_PROMPT.md`.

GitHub connector inspection found `Riteshhai/Retina-Opthalmology`, selected by
the user as the target. Its main branch contains only README.md. No existing
QSART implementation was found in the connected repository inventory.

## Reusable components

- `qsart/deepdrid.py`: official metadata parser, eye-specific DR labels,
  patient-overlap audit. Initially metadata-only; does not check image files.
- `qsart/policy.py`: deterministic three-action threshold research policies and
  calibration-only grid tuning. Budget constraints are checked during tuning,
  but were not enforced on a shifted evaluation population.
- `qsart/cli.py`: independent baseline tuning and patient-leakage rejection.
- Starter tests, literature matrix, protocol and manuscript outline.

The starter's `routine` outcome treats every such image as an auto-clearance;
it is not equivalent to the requested provisional five-grade AUTO_SCREEN.
Keep this legacy research benchmark separate from the new model-backed API.

## Missing at audit

No retinal dataset images, trained checkpoints, real training runs, model-backed
API, calibration artifacts, or measured real-image deployment results are present.
All built-in starter scores are explicitly synthetic software fixtures. Prior
literature assertions require primary-source verification before reuse.

Python 3.11 and Git are present; torch/torchvision were absent. Docker CLI exists;
daemon availability and container execution require verification. A dedicated
`.venv` is being established. Do not infer that a CLI executable means deployment
has been verified.

## Implementation decisions

Preserve the small `qsart` package layout instead of moving working imports.
Add data, preprocessing, training, inference, calibration, exact cohort-budget
allocation, FastAPI and a Streamlit client. Synthetic fixtures remain tests;
production startup rejects test checkpoints. Official validation is reserved as
a held-out population; training, checkpoint selection and calibration use
patient-separated subdivisions of official training data.

## Acceptance status

PARTIALLY COMPLETE. The original source is a software policy proof of concept,
not a trained retinal screening system. See TASKS.md and PROGRESS.md for evidence
as implementation proceeds, and docs/research for source verification.
