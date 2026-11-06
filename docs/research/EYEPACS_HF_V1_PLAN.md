# User-selected EyePACS mirror cache — 6 November 2026

The user requested online acquisition from
[bumbledeep/eyepacs](https://huggingface.co/datasets/bumbledeep/eyepacs). Pin revision
`3e94599d7e41d9992a8362cc0467eb397e11210f`, fourteen Parquet shards,
6,535,643,877 bytes and 35,108 declared rows. Preserve exact file sizes/LFS SHA256
and the pinned dataset-card hash in configs/eyepacs_hf_v1.json.

This is EyePACS with disease labels, not a release containing EyeQ quality labels.
The mirror card describes cropping, resizing up to 1024 pixels and removal of
nearly black/unmaskable images. It claims MIT and links a resized Kaggle source;
the card alone does not independently establish the underlying original-image
terms. Record the user-selected community source and these limits. No account,
credential or access gate is supplied, inferred or bypassed.

A bounded Parquet metadata probe fetched 66,599 bytes and found the original
numeric-subject/eye JPEG names in image.path. Embedded image bytes were not read
or decoded by that probe. The original-name field makes a join technically
possible; no full-cohort match or quality-validation result is assumed in advance.
The initial source/card/schema/size findings were inspected before this plan.
This is an engineering acquisition audit, not a blind neural experiment.

## Cache and integrity behavior

Default CLI prints a space/source plan and downloads nothing. Explicit --download
fetches the pinned card and files once under ignored project data. Two workers
are the default; --workers supports one, two or four concurrent files. Bounded
32-MiB requests and twelve consecutive network retries tolerate long-response
disconnects. Each file resumes only from a partial bound to its
repository, revision, exact size and SHA256. Require the exact HTTP resume range,
reject size/checksum changes, preserve differing local bytes and never overwrite
a completed cache file. Restrict redirects to HTTPS Hugging Face/CDN domains;
fail on access gates without switching sources. No remote dataset code is run.
Store a local receipt. Reuse verified files on future runs; app startup never
downloads this data. Reserve two GB beyond the required cache space.

Only project data/artifact roots may receive cache/audit outputs. Read-only
Parquet projections inspect image.path, label_code and label; they do not decode
JPEGs. Require unique canonical original names, DR codes 0–4 and exact matching
label names. Compare the full mirror cardinality to the pinned 35,108 rows.

## Annotation compatibility and remaining gates

Join by exact original name to the existing immutable seven-file EyeQ metadata
release. Retain Good/Usable/Reject and source DR separately. Check all shared DR
labels for disagreement and report matched/missing counts by original quality
class. Preserve the private join; never assign missing labels or infer matches
from row order. This manifest is external-candidate only and training-ineligible.

Resizing/cropping/row removal can affect image-quality interpretation. A complete
identifier join would not establish original pixel equivalence, valid transported
quality truth, legal-source verification or independent-patient/pixel separation.
Keep those readiness fields false. No model fitting, predictions, test consumption,
source partition edits or production change occurs during acquisition.

After successful transfer, independently verify every shard hash, schema, row and
original-name/DR/quality join; preserve all earlier report/master/checkpoint/input
bytes. Publish aggregate acquisition evidence only. Photographs, Parquet payloads,
source IDs and private joins remain local. Continue original-EyeQ acquisition
when available; never silently replace the frozen original-image adapter.

Primary implementation reference: [Hugging Face download documentation](https://huggingface.co/docs/hub/datasets-downloading).
Software tests cover verified cache reuse, exact resumption, ignored/wrong ranges,
changed markers/checksums, trusted redirects, access failure, metadata-only reads
with deliberately invalid synthetic JPEG bytes, missingness and label conflicts.
Run the full genuine-model/conformal suite, Ruff and available deployment checks.
