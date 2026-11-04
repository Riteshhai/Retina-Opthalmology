# IDRiD source confirmation and overlap review v1 — results

**Later execution:** [locked external evaluation](IDRID_EXTERNAL_V1_RESULTS.md)
has now consumed model outcomes. This preserved data-review stage did not perform
model inference; its population choices remain fixed.

**VERIFIED COMPLETE:** bound source confirmation, fixed candidate pixel review and
label-independent population freeze. External model evaluation is **NOT STARTED**.
Execution used clean `d1c5968189f8bd36508b5cbb4c35220af5f0bf6b` under the
[declared protocol](IDRID_REVIEW_V1_PROTOCOL.md). Measured review SHA256:
`f21a3cd6109030c6c9da96710b7af0e6a89a00365afbfc76af9cb539abdf15f7`.

## Source confirmation and preservation

The user confirmed: “Yes, official IDRiD/IEEE DataPort release”. The separate
[source receipt](../../reports/idrid_data_v1/source_confirmation.json) binds the
original audit, manifest and image-content hashes. This statement does not verify
an official archive checksum. The original audit and its audit-time unconfirmed
status stay byte-identical; API/UI now expose current confirmation and historical
status separately. Missing, stale or malformed receipts cannot upgrade provenance
or clear the overlap gate.

## Fixed candidate review

All 8 internal and 3 APTOS candidate pairs were checked against the frozen image
SHA256 values. Comparisons distinguish source bytes, EXIF-oriented original RGB
pixels and descriptive canonical 224px geometry. Canonical equality, MAE, RMSE and
correlation are not a clinical identity test and never determine population masks.

| Pair family | Pairs | Byte-identical | Identical original RGB with different bytes | Unresolved fingerprint candidates |
| --- | ---: | ---: | ---: | ---: |
| internal | 8 | 6 | 0 | 2 |
| cross_source | 3 | 0 | 0 | 3 |

All three cross-source pairs remain unresolved: neither original RGB pixels nor
canonical 224px arrays are identical. This does not prove different patients/eyes or
rule out re-encoding, cropping and related views. The conservative rule retains
all three candidates in the flagged set regardless of these descriptive metrics.
No pixel-similarity threshold was adjusted after inspecting results.

There are **3 byte-identical internal pairs with disagreeing DR labels**.
Across all 8 internal candidates,5 pairs have DR disagreement and 2 have DME-risk
disagreement. All four label-conflicting components remain flagged. No majority
vote, clinical adjudication, quality-label substitution or source-grade change
occurs. Pixel identity cannot determine the correct clinical grade.

## Frozen sensitivity populations

The full original manifest and all 516 observations are preserved. The new private
population table adds membership flags without changing original columns. Every
row remains external_holdout with eligible_for_training=False.

| View | Images | Fingerprint components | Label-conflicting components |
| --- | ---: | ---: | ---: |
| full_source | 516 | 508 | 4 |
| cross_source_filtered | 513 | 505 | 4 |
| cross_source_filtered_representatives | 505 | 505 | 4 |

The conservative view flags every IDRiD component touching an original APTOS
candidate. Representative selection takes the minimum SHA256 of a fixed namespace
plus source identity within each retained component. Neither rule uses labels,
row order, pixel-comparison outcomes, model predictions or accuracy. All four
ambiguous components remain explicit even in the representative view; choosing
one source label does not resolve their annotation uncertainty.

These views are sensitivity populations, not patient cohorts or a claim of complete
decontamination. A later locked evaluation must report full/filtered populations,
state the unknown patient relationships and analyze ambiguous source annotations.
No population is selected because it gives favorable classifier performance.

## Verification and current application

Independent reconstruction reproduced all 11 pair comparisons, every 516 membership
decision, original fields/labels, reference/source hashes and source-receipt binding.
All 20 previously preserved master/production/APTOS artifacts matched; the frozen
IDRiD audit and private inputs also matched. Original per-image paths, identities,
pixel statistics and memberships stay local/ignored under artifacts/idrid_review_v1.

The full suite passed **476 tests**, zero failures/skips, 77.12s with real trained
production models and conformal integration. Ruff and Compose configuration passed.
Tests cover stale receipt rejection, historical status retention, component-wide
flags, row-order/label-independent representatives and decoded-vs-byte identity.

Actual native API and Streamlit component checks passed after restart: 14 evidence
records, current source confirmation, retained overlap warning, all population
counts and zero external predictions. Production card/sidecar still match. Real
JPEG single/batch/TTA/readiness/simulator checks passed. New browser rendering and
container execution remain unverified; the Docker engine is currently unavailable.

## Remaining requirements

Source-origin confirmation is complete. Clinical-grade contradictions and three
cross-source perceptual candidates remain unresolved; this report does not claim
an independent-patient cohort. Predeclare models/checkpoints, scoring, full and
sensitivity analyses and ambiguity handling before any external model outcomes.
Stronger development optimization remains separate. Useful validated triage and
clinical utility remain unestablished, and formal manuscript drafting stays deferred.

Exact evidence: [review aggregates](../../reports/idrid_review_v1/data_review.json),
[independent reconstruction](../../reports/idrid_review_v1/independent_audit.json),
[verification](../../reports/idrid_review_v1/verification.json), and preserved
[original data audit](IDRID_DATA_V1_RESULTS.md). Overall project **IN PROGRESS**.
