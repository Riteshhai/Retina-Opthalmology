# EyeQ annotation/access audit v1 — 5 November 2026

This is a metadata/readiness milestone. No EyeQ photograph was acquired, decoded
or predicted during preparation. Author CSVs were inspected before this protocol;
their discovered counts and table disagreements are disclosed, not blind outcomes.
No neural experiment or independent-quality validation is declared complete here.

## Pinned sources and scope

Pin HzFu/EyeQ revision `1dcfabb08cd31045de1a4a3ee82e48f6833697aa`. Fetch only
the README, two official CSVs and four preprocessing/loader/main code files.
Bind SHA256 plus Git-blob SHA1, preserve local copies and record verification time.
Do not download the example photographs or any mirror. Code is read as source
evidence, not run, copied into QSART modules or used to license source photographs.

Primary references: [pinned author repository](https://github.com/HzFu/EyeQ/tree/1dcfabb08cd31045de1a4a3ee82e48f6833697aa),
[paper v4](https://arxiv.org/abs/1907.05345v4), [official EyePACS source](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=test.zip.004).
The indexed official image page exposes a sign-in/rule-acceptance gate; direct text
extraction of rules/data returned no readable body. No account, credential or
acceptance is supplied or inferred. Preserve this access limit and the distinction
between code, annotations and photograph terms. DDR's pinned license wording is
a source comparison only, not an alternative dataset acquisition or quality claim.

## Exact schema and independent fields

Require exact CSV columns `Unnamed: 0,image,quality,DR_grade`, serialized row index
0..n-1, complete integer quality0..2 and separate source DR0..4. Original names
must follow the official EyePACS numeric-subject/left-or-right `.jpeg` convention.
Reject traversal, aliases, missing values, duplicate image IDs and changed pinned
bytes. Keep source rows/splits, eye grouping, three quality codes/names and source
DR grades private. Do not convert source indices into patient identities.

Author code orders classes Good,Usable,Reject (0,1,2). Source definitions distinguish
quality from disease; an existing source DR grade is not erased by Reject quality.
No DeepDRiD `quality_grade`, natural-defect label or generic training DR label is
created. Both source splits become external-only candidates, training-ineligible.
Filename subject keys keep known eyes together but do not prove cross-source
patient/pixel independence. Record per-split groups, singleton/pair counts, eye
counts, quality/DR matrices, discordant-eye counts and cross-split group overlap.
Differences between eyes are image-specific annotations, not automatic errors.

## Known preparation findings and harmonization boundary

Observed CSV totals:12543 train and16249 test. Train quality counts8347/1876/2320;
test8471/4558/3220. The README's train usable count1896 disagrees; paper v4 corrects
that value to1876. Test good/usable totals in the paper/README differ by one from
the pinned CSV. Retain these disagreements without relabeling or dropping rows.

Expose only explicit named binary sensitivity views: Reject versus non-Reject,
or Usable-or-Reject versus Good. Their population counts are metadata diagnostics,
not a selected harmonization or equivalence to DeepDRiD poor quality. No endpoint,
model, temperature, threshold or preprocessing protocol is selected from EyeQ
outcomes. A future external-evaluation protocol must resolve these choices before
predictions and after source/terms/image/overlap readiness is established.

## Future local-image inventory

The optional private inventory accepts original JPEG names in a flat root or
`train`/`test` subfolders. Fail on escaping paths or ambiguous copies; do not
silently use the authors' preprocessed PNG naming convention. Verify actual JPEG
format, dimensions, decode, SHA256 and the established retinal pHash without
changing source bytes. Report missing/invalid/exact-duplicate files privately.
Inspection does not infer download provenance, terms acceptance, cross-source
overlap or readiness for model evaluation. It never fits or predicts a model.

The default public audit has zero images decoded and false pixel/independence/
harmonization readiness. Its official whole-archive listing is88.29GB, not a
verified EyeQ subset or unpacked footprint. Capture project-volume free bytes
and state the measured capacity comparison; do not start a full image download.

## Execution and verification

Default `python scripts/audit_eyeq_data.py` verifies local pinned metadata only.
`--fetch-metadata` retrieves only the closed, hash-bound public file list.
After tested source/config/protocol are committed clean, `--execute` freezes an
aggregate data_audit.json and private external annotation/group manifests. Refuse
overwrites. Optional `--image-root` produces a new private inventory under
artifacts, outside original-image roots; it cannot rewrite the frozen public audit.

Independently reconstruct source object hashes, all CSV/schema/count/group/matrix
and binary-view results with standard-library CSV/group operations, not the new
adapter; verify private manifest digests, publication privacy, old measured bytes
and explicit absence of prediction/quality-accuracy claims. Verify API/UI readiness
labels and unchanged genuine production inference. Run pytest/Ruff/Compose and
available runtime checks. No container proof without a working engine. Preserve
both exact master copies, all historical studies and the deferred manuscript.
