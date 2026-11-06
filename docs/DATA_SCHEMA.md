# Verified DeepDRiD data contract

Source: [official DeepDRiD repository](https://github.com/deepdrdoc/DeepDRiD), pinned to
`56d8af71319803150d0e7b90533e0918c7bf85ee`. The local annotation CSVs, original
label-definition `Readme.docx`, and repository `LICENSE` are in ignored
`data/raw/deepdrid/metadata/`. Their source URLs and SHA256 hashes accompany the pilot.
The repository supplies CC BY-SA 4.0 terms; acknowledge Liu et al., *Patterns* (2022),
DOI [10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512).
Do not commit local images or patient manifests.

The downloaded training CSV has 1,200 images / 300 patients. Official validation
has 400 images / 100 different patients. Typical observations are four images per
patient: two views per eye. Each image is a training observation, while **the
authoritative CSV `patient_id` is the indivisible split group**. Image names alone
are never used to invent patient identities. Namespace `deepdrid_` is prepended to
the supplied patient ID. Left/right DR labels are eye labels repeated across views;
the more severe `patient_DR_Level` is retained for traceability, never substituted
as the image's DR label.

| Manifest field | Meaning |
|---|---|
| `image_id` | Unchanged official image identifier |
| `image_path` | Resolved absolute local path; a missing path remains explicitly missing |
| `dataset_source` | `DeepDRiD_regular_fundus` |
| `subject_id_original`, `subject_id` | Supplied patient ID and dataset namespace |
| `eye` | Verified conventional filename side, otherwise missing and quarantined |
| `dr_grade` | Selected eye's 0–4 DR grade; never a fabricated diagnosis |
| `quality_grade` | Official overall-quality code: 0 poor, 1 good |
| `clarity`, `field_definition`, `artifact` | Unchanged verified ordinal component codes |
| `official_split`, `split` | Source split and QSART experiment role |
| `annotation_status` | `verified` or `quarantined_eye_or_grade_conflict` |
| `exists`, `readable`, `image_status` | Actual filesystem/Pillow integrity checks |
| `width`, `height`, `sha256`, perceptual fingerprints | Decoded image shape and duplicate evidence |
| `source_annotation`, `source_annotation_sha256` | Source CSV provenance |
| `original_*` | Every original CSV column, without corrected eye labels |
| `eligible_for_training` | Readable image with verified annotation; excludes quarantine |

Official label meanings come from
[the source Readme.docx](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/regular_fundus_images/regular-fundus-training/Readme.docx).
Overall-quality 0 means insufficient quality for diagnosing retinal diseases; 1
means sufficient quality according to the dataset graders. These are dataset
annotations, not QSART clinical certification. DR codes 0/1/2/3/4 correspond to
no apparent DR / mild NPDR / moderate NPDR / severe NPDR / PDR. The Readme also
defines 5 as ungradable, but none occur in downloaded train/validation eye grades;
future such rows are quarantined, never forced into the five disease classes.

Clarity and field definition use `{1,4,6,8,10}` with larger values better.
Artifact uses `{0,1,4,6,8,10}` with larger values worse; 0 means no artifact.
These are ordinal codes with unequal spacing, **not zero-based class IDs and not
continuous probabilities**. Optional categorical heads must save an explicit
code-to-class mapping. No illumination annotation is present in these CSVs, so
the pipeline does not fabricate one.

Eleven official training rows have nonconventional view suffixes (3/4) and
filename-side conflicts with the populated eye-label column. Four patients are
affected: 56, 77, 164, 167. These rows preserve original data, with missing normalized
DR/eye labels and quarantine status. The pilot excludes all four affected subjects.
The strict historical `qsart.deepdrid.normalize_labels` continues rejecting these
rows; the new manifest parser reuses its normalization only for verified rows.

`train`, `calibration`, and `validation` are patient-disjoint partitions drawn
exclusively from official training. Subject allocation is deterministic at seed
2026 and stratified by maximum verified eye grade. Official validation is a locked
`holdout`, usable only after model fitting and selection. No official challenge
test files are acquired, no holdout thresholds are tuned, and official validation
is not claimed as independent external-domain validation.

The initial acquisition is a deliberately enriched engineering pilot: ten
official training patients and three official validation patients per maximum
patient eye-grade, all four available views. It yields 50 training-source patients
and 15 holdout patients. It does not reflect screening prevalence. The selection
JSON is fixed before training; its deterministic sampling and image URLs/hashes
are saved beside the manifest. A successful pilot establishes executable plumbing,
not a research-performance or clinical-safety conclusion.

Source JPEGs remain untouched. Shared preprocessing applies EXIF orientation,
RGB conversion, conservative non-dark border cropping, aspect-preserving square
padding, and configurable bilinear resizing. It does not enhance lesions, infer
fundus presence, or assert image gradability. The same helper is imported by
training and inference.

Run `python scripts/prepare_data.py --download-pilot --out data/processed/deepdrid/pilot_manifest.csv`
for reproducible acquisition, or supply `--root`, `--train-csv`, `--validation-csv`
for an existing local dataset. `QSART_DATA_ROOT` overrides the default image root.
Use `--config configs/data.yaml` to load data paths, split fractions, and the seed;
explicit CLI values override the configuration, with `QSART_DATA_ROOT` overriding
the configured root. Source-annotated images remain in the ignored local folders.
`python scripts/validate_data.py data/processed/deepdrid/pilot_manifest.csv`
reopens every image, recomputes hashes, and returns nonzero on missing/corrupt
images, unknown labels, subject leakage, duplicates, or unresolved cross-partition
perceptual duplicate candidates. Missing images may be audited but are never
silently declared ready for training.

The initial frozen pilot CSV predates addition of the `phash` column. Its original
whole-image `dhash` values yield numerous circular-border false candidates;
revalidation computes `central_green_dct_phash_v1` on reopened pixels and saves
`pilot_manifest.verified_audit.json`, tied to the unchanged CSV SHA256. Read that
audit for the actual pilot readiness decision. Newly generated manifests include
both fingerprints. The stronger fingerprint crops black borders and uses the
central green channel to reduce outline-driven collisions; it is a duplicate
review aid, not a guarantee that all related images have been identified.
