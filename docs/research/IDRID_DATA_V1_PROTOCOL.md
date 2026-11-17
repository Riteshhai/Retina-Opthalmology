# IDRiD external-data readiness v1 — declared 4 November 2026

This step validates the newly discovered local disease-grading package and freezes
an external-only manifest. It performs no model inference, fitting, calibration,
threshold tuning or evaluation of classifier outcomes. IDRiD remains reserved.
The completed APTOS studies and production artifacts stay byte-identical. Formal
manuscript work remains deferred.

## Source and annotation contract

Primary sources: [official data description](https://idrid.grand-challenge.org/Data/)
and [original dataset descriptor](https://www.mdpi.com/2306-5729/3/3/25).
The official disease-grading release describes 516 images, with 413/103 original
training/testing images. Both original partitions are external to QSART; neither
becomes a QSART fitting or calibration split. The reported 455-image base-paper
cohort is not reproduced by silently discarding 61 release images.

Read only `B. Disease Grading`, its included attribution/license, two grading CSVs
and corresponding original JPG directories. Ignore segmentation/localization
copies. Require exactly the three named fields after trimming header whitespace;
allow only entirely blank trailing `Unnamed:` columns from spreadsheet export.
Reject unexpected populated fields, invalid/duplicate/unsafe image stems,
noninteger/out-of-range labels, missing/extra images, unreadable images and image
paths resolving outside the supplied root. Production execution requires exactly
413/103 rows. Small generated software fixtures may use the same adapter schema.

Preserve source strings and annotation-byte hashes. Map `Retinopathy grade` to
DR grades 0–4 without alteration; retain `Risk of macular edema` separately as
DME risk 0–2. Never treat DME risk as DR severity or an image-quality target.
The package provides no authoritative patient/eye identities or quality labels.
Serial names and fingerprint components do not establish patient identity.
Record included CC BY 4.0 attribution/license byte hashes; file presence/layout
does not authenticate the download. Record user-confirmed official provenance
only if the user actually supplies it; otherwise state unconfirmed local origin.

## Fixed integrity and overlap audit

Before real execution commit the protocol, adapter, runner and tests. Decode
every grading JPG; retain dimensions, SHA256 and the existing central-green DCT
pHash. Use fixed Hamming distance <=2, identical to the established APTOS audit.
Form all within-IDRiD exact/perceptual connected components without relabeling,
quarantining or discarding external images. Flag label disagreements as audit
findings. Identity is namespaced by original partition to prevent stem collisions.

Check every external image against these four frozen fingerprint manifests:
APTOS retained 3,514, APTOS quarantine 148, DeepDRiD development 1,184, and consumed
DeepDRiD confirmation 340. Bind each input manifest by SHA256 and source dataset.
Include all source APTOS partitions and quarantined images, not just fitting rows.
Require complete valid fingerprints and verify repeated reference identities agree.
Report reference rows, unique identities and exact/perceptual candidate counts by
reference dataset, without exporting identities. This check uses saved verified
reference fingerprints; it is not a fresh reinspection of every reference JPEG/PNG.

Every match remains a private review candidate. Never remove observations based
on external labels or the desired model result. Any detected cross-source overlap
blocks an unqualified independent-cohort claim; a later evaluation protocol must
resolve exclusions, present full/overlap-sensitive populations and acknowledge
that the external label distribution has now been inspected. No visual same-patient
identity or complete decontamination claim follows from a negative pHash audit.

## Outputs and completion criteria

Store image identities, absolute paths, fingerprints, component membership and
pairwise candidates ONLY under ignored `data/manifests/idrid_v1` or `artifacts`.
All rows use `external_holdout`, `eligible_for_training=False`, and an explicit
reserved/not-inferred status. Existing generic training readers must reject them.
Public aggregate report `reports/idrid_data_v1/data_audit.json` records source
status, label counts, dimensions, duplicate/overlap counts, hashes, source commit
and limitations. Refuse overwriting completed output bundles. No model or set
coverage metric belongs in this data-readiness report.

Independently reconstruct source/manifest/pair hashes, label mappings, components,
all fixed pairwise distances and aggregate counts from the saved private outputs.
Run meaningful fixture tests and Ruff; preserve all prior master/results/model
bytes. Update the current backlog and source/README documentation with the actual
audit result, provenance uncertainty and remaining external-evaluation gate.
