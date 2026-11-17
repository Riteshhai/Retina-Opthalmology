# Local IDRiD disease-grading data

**Current status:** [locked external evaluation](research/IDRID_EXTERNAL_V1_RESULTS.md)
has consumed IDRiD model outcomes. The audit/review snapshots below describe their
earlier no-inference stages. Never use this data for model/calibration/policy fitting,
or describe later reuse as fresh confirmation. Source grades and original files
remain preserved; official/internal APTOS test remains reserved.

The existing package is discovered under the major-project folder. Its disease-
grading directory contains `1. Original Images`, `2. Groundtruths`, `LICENSE.txt`
and `CC-BY-4.0.txt`. Segmentation/localization image copies are not added to the
external grading population.

`scripts/prepare_idrid.py` resolves `QSART_IDRID_ROOT` from the environment, then
the ignored local `.env`. Relative values resolve from QSART; the fallback is
`../idrid/B. Disease Grading`. The source root stays local. It never downloads
data or changes images. The included attribution/license and official source
links are recorded, with download-origin status stated separately.

The [completed audit](research/IDRID_DATA_V1_RESULTS.md) freezes all 516 observations
under ignored `data/manifests/idrid_v1`. Completed or partial output bundles refuse
overwrite. Do not rerun the original audit into these paths. For a new declared
data audit, use a new output identity, clean committed source, and its exact
`--source-commit`; `--root` overrides discovery. Only use
`--source-confirmed-official` for an actual user confirmation of that source.

Open Experiment Explorer → Dataset IDRiD to inspect aggregate readiness, original
partition counts, source status and overlap warnings. This is data evidence, not
external model performance. Individual images, identities, source paths, hashes,
pair tables and manifests remain private and excluded from Git.

All rows are `external_holdout` with `eligible_for_training=False`. Generic
training readers reject the manifest. Source label distributions were read for
data validation; no external model outcome had been observed at that data-freeze stage. Resolve provenance
and candidate overlaps before claiming an independent cohort. Predeclare scoring
and analysis before any later external inference, and keep this data out of model,
temperature, conformal-threshold and policy fitting.

## Current source and candidate review

The user has confirmed the official IDRiD/IEEE DataPort source. The bound
source_confirmation.json receipt preserves the original audit-time status; the
current API/UI display the confirmation and retain the overlap warning.

The [completed review](research/IDRID_REVIEW_V1_RESULTS.md) freezes full 516,
filtered 513 and representative 505 views under ignored artifacts/idrid_review_v1.
No source labels or original rows were changed. scripts/review_idrid.py records
the fixed source/pixel checks and refuses to overwrite this completed bundle.
Clinical-grade conflicts and cross-source candidates remain explicit; external
model evaluation needs a locked protocol and ambiguity-sensitive interpretation.
