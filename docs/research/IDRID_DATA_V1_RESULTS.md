# IDRiD external-data readiness v1 — results

**VERIFIED COMPLETE:** source-layout, JPEG integrity and fixed fingerprint audit.
**NOT STARTED:** external model evaluation. All 516 images remain reserved;
no prediction, model fit, temperature fit or conformal-score fit used this cohort.
The [protocol](IDRID_DATA_V1_PROTOCOL.md) preceded execution from clean commit
`d597daa553b58c57f8e96f1b33a2f24bf8a9e4f8`. Aggregate SHA256: `b31891fc14c5c52ab1e638821a5e014762e20d2add6a936ab18d093ef165f88a`.

## Provenance and population

Download-origin status: **local_package_download_origin_unconfirmed**. Included attribution
and CC BY 4.0 license bytes were recorded. Local layout/license matching does not
authenticate a download. User source confirmation is kept distinct from image
integrity and official archive authentication.

Primary sources are the [official IDRiD data description](https://idrid.grand-challenge.org/Data/)
and [original dataset descriptor](https://www.mdpi.com/2306-5729/3/3/25). They describe
516 graded images in 413/103 original training/testing partitions. Both partitions
are external holdout for QSART. The base paper's reported 455-image cohort cannot
be reproduced by dropping images without an audited rationale.

| Original partition | Images | DR 0 | DR 1 | DR 2 | DR 3 | DR 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_test | 103 | 34 | 5 | 32 | 19 | 13 |
| official_train | 413 | 134 | 20 | 136 | 74 | 49 |

DR grades are preserved as 0–4. DME risk remains a separate 0–2 annotation;
it is never substituted for DR or image quality. Entirely blank spreadsheet
export columns are tolerated; populated unknown fields and invalid source labels
fail closed. Serial image names are namespaced by original partition. No patient,
eye or quality labels are invented.

## Integrity and overlap

All 516 source JPEGs decoded successfully and were bound by
SHA256, dimensions and central-green DCT pHash. Fixed Hamming distance <=2
produced 8 internal pairs (6 byte-exact),
7 duplicate components and
4 label-conflicting components.
All 516 observations and source labels are retained; no exclusions or
outcome-driven relabeling took place.

| Reference source | Rows | Unique identities | Candidate pairs | Byte-exact pairs |
| --- | ---: | ---: | ---: | ---: |
| APTOS_2019 | 3662 | 3662 | 3 | 0 |
| DeepDRiD_regular_fundus | 1524 | 1524 | 0 | 0 |

The reference pool contains all 3,662 APTOS images, including the 148 quarantined
observations, plus DeepDRiD development and consumed confirmation images.
There are **3 cross-source candidate pairs** across
3 IDRiD images, including 0 byte-exact pairs.
Detected-overlap independence gate: **blocked**.

These are fingerprint candidates, not proof of a common patient or eye. A negative
audit cannot exclude all related views, re-encodings or images outside the checked
reference pool. Any candidate requires review before an independent-cohort claim.
Saved reference fingerprints were checked against frozen manifests; reference
JPEG/PNG files were not freshly reinspected in this step. Individual identities,
paths, hashes, pair rows and component membership remain private and ignored.

## Verification and application

Independent reconstruction decoded, rehashed and recomputed all 516 source pHashes
using a separate scipy DCT implementation. It reproduced every component and pair
over 132,870 within-cohort and
2,675,976 cross-source distances. Source label/DME
mapping, annotation/license/manifest hashes and 20 preserved
master, production and earlier research artifacts matched.

The full suite passed **456 tests**, zero failures/skips, 75.91s with actual
production-model/conformal integration enabled; Ruff and Compose configuration
passed. The first new API fixture assertion was corrected to query discovery
metadata through the documented route, then the full suite was rerun.

Actual native research HTTP and Streamlit AppTest show the reserved cohort,
original-partition counts, provenance and overlap warnings, and zero external
predictions. Original production card/sidecar match. These component checks are
separate from new container execution or browser-rendering evidence.

## Remaining gates

Source provenance must be resolved transparently. Review any overlap candidates
under a fixed label-independent rule before external evaluation. Predeclare the
models/checkpoints, population, preprocessing, temperatures, uncertainty seeds,
alphas and analysis before computing classifier outcomes. External labels have
now been inspected for data validation and aggregate distribution reporting;
no external performance result has been observed. Do not later describe this as
an untouched-label cohort or tune models/policies on external outcomes.

The four-epoch APTOS pilot remains limited; stronger optimization and a locked
independent evaluation are separate requirements. Useful validated triage and
clinical utility remain unestablished. Overall project IN PROGRESS; formal
manuscript drafting remains deferred.

Exact evidence: [aggregate audit](../../reports/idrid_data_v1/data_audit.json),
[independent reconstruction](../../reports/idrid_data_v1/independent_audit.json),
[verification](../../reports/idrid_data_v1/verification.json). The external manifest
uses `external_holdout` and `eligible_for_training=False`; generic training
readers reject it. Completed audit bundles refuse overwriting.
