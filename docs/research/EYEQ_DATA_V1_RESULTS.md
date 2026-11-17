# EyeQ annotation and access audit v1 — 5 November 2026

**VERIFIED COMPLETE for metadata/access auditing. EyeQ image/model validation
remains IN PROGRESS.** Executed from clean source
`9f7f22f79362090039369f8a34625d326189c6e2` under MASTER_SPEC_v1.0. No EyeQ
photograph was downloaded, decoded or predicted, and no classifier, temperature,
conformal sidecar, policy or production model was changed.

## Verified source and labels

Seven public author files are bound to EyeQ revision
`1dcfabb08cd31045de1a4a3ee82e48f6833697aa`, SHA256, Git-blob SHA1 and an append-only
local verification receipt. The closed download list contains the README, two
label CSVs and four source files used to check preprocessing/class semantics.
Author code was read, never executed or imported into the QSART implementation.
Example photographs, author predictions and weights were not fetched.

Primary sources are the [pinned author repository](https://github.com/HzFu/EyeQ/tree/1dcfabb08cd31045de1a4a3ee82e48f6833697aa),
[paper v4](https://arxiv.org/abs/1907.05345v4) and
[official EyePACS image page](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=test.zip.004).
The serialized CSV header begins with a blank index field; pandas names that
field `Unnamed: 0`. This distinction explains the protocol's parsed-schema name.
The independent CSV verifier checks the literal blank header and sequential indices.

| Original author split | Images | Filename subject keys | Good (0) | Usable (1) | Reject (2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 12,543 | 8,614 | 8,347 | 1,876 | 2,320 |
| test | 16,249 | 12,048 | 8,471 | 4,558 | 3,220 |
| total | 28,792 | 20,662 | 16,818 | 6,434 | 5,540 |

All image identifiers are unique. No filename subject key appears in both author
splits. Train contains 4,685 singleton keys and 3,929 eye pairs; test contains
7,847 singleton keys and 4,201 pairs. Quality labels differ between eyes in
372 train pairs and 629 test pairs. These image-specific differences are retained.
Source DR grades remain separate, including on Reject images. Filename grouping
preserves known eye pairs; it cannot establish cross-source patient/pixel independence.

Both author splits are external-only and training-ineligible here. Good/Usable/
Reject are preserved; source code 0 is Good, whereas DeepDRiD overall quality 0
means poor. Two explicit binary sensitivity views expose population counts only.
Neither is selected as the external endpoint or used to create generic training labels.

## Table disagreements and access limits

The pinned README displays 1,896 train Usable images, versus 1,876 in the CSV;
paper v4 displays the matching 1,876. For test Good/Usable, the README/paper display
8,470/4,559, versus CSV counts 8,471/4,558. The audit retains all four comparisons
and displays the three nonzero disagreements. No rows are dropped or relabeled,
and no undocumented explanation is assigned to the discrepancy.

The author README explicitly describes a code license; it does not replace
underlying image terms or settle a separate unrestricted annotation permission.
The official indexed Kaggle page indicates sign-in/rule acceptance. Direct rules/
data extraction returned no readable body, so accepted terms are not asserted.
Pinned DDR README/LICENSE were reviewed only to distinguish its dataset and code
license statements; no DDR image or quality-label substitution occurred.

The official indexed listing is 88.29 GB for the entire competition. Audit-time
free project-volume space was 76,828,942,336 bytes, about 76.83 decimal GB. The full
archive exceeds that space even before unpacking. The EyeQ subset and unpacked
footprints are unmeasured. No full download, dataset deletion or access bypass
was attempted. See [local preparation steps](../EYEQ_SETUP.md).

## Reconstruction and actual execution

The standard-library auditor independently reconstructs all 28,792 source rows and
private manifest/group rows, 20,662 subject keys, both split profiles, 30 quality/DR
cells, four binary population views and four table comparisons. All seven source
objects match the independently saved GitHub tree. It verifies source/config/
protocol bytes at the execution commit, the receipt, two comparative terms files,
public privacy and all 271 historical/master/production artifacts. Public reports
contain aggregate evidence only; identifiers, local paths and downloaded source
copies remain in ignored local storage.

The first execution at `0c6124f6779f9bf21faef5f91934f684bad60c04` stopped before
publication because a legacy guard mistook HTTPS for a drive path. Its partial
private manifests are preserved and byte-identical to the completed exports.
The corrected guard has explicit path/credential/host/private-key regression
checks. The independent auditor was corrected before export to read the literal
blank CSV header and explicit UTF-8 on Windows; source data/results were unchanged.

All **558 tests passed**, zero failures/errors/skips, with genuine trained-model/
conformal integration; JUnit suite time 102.731 seconds. All **22 targeted tests
passed** in 11.840 seconds. Ruff, Compose configuration and whitespace checks pass.
The [source CI run](https://github.com/Riteshhai/Retina-Opthalmology/actions/runs/42080027323)
passed at the exact execution source. Full and targeted test XML remain local.

Actual native HTTP verifies JPEG single/batch/TTA, matched production card/sidecar,
readiness, simulation and overload. Live-API Streamlit AppTest verifies 19 records,
two source-split rows, three quality categories, three table disagreements and
explicit zero-image/incomplete-readiness captions without an EyeQ accuracy claim.

Docker initially lacked its Linux engine pipe, then became available during this
session. The unchanged application source was actually rebuilt and run in healthy,
nonroot, read-only CPU containers on temporary ports 28000/28501. The same real
inference/conformal/simulator/overload checks and 19-record metadata explorer pass.
All **65 runtime source/configuration files** match host bytes; image identity,
health and installed dependencies are recorded. This is current container proof.
AppTest ran against actual container HTTP, with the container dashboard source
separately hash-matched; no new interactive browser-rendering check is claimed.
Temporary verification containers were removed after checks. The native
application remains available on 18000/18501.

## Evidence and remaining work

Aggregate report SHA256:
`e0941a2319941e28052c614b68a618d5fe9ed850f71cc7b3c3e30e6b7755b645`.

- [Frozen metadata/access report](../../reports/eyeq_data_v1/data_audit.json)
- [Independent reconstruction](../../reports/eyeq_data_v1/independent_audit.json)
- [Combined verification](../../reports/eyeq_data_v1/verification.json)
- [Native deployment](../../reports/eyeq_data_v1/native_deployment.json) and [explorer](../../reports/eyeq_data_v1/native_registry.json)
- [Container runtime](../../reports/eyeq_data_v1/container_runtime.json), [deployment](../../reports/eyeq_data_v1/container_deployment.json) and [explorer](../../reports/eyeq_data_v1/container_registry.json)

Before EyeQ predictions: acquire official source photographs with locally recorded
source/accepted terms and a feasible storage plan; inventory original JPEGs; check
pixel overlap against all previously used datasets; and lock the external cohort,
quality harmonization, model states, preprocessing and evaluation. The private
inventory adapter is implemented and fixture-tested; real EyeQ pixels remain
unavailable. No new quality accuracy, clinical recapture benefit, independent
patient cohort or validated triage utility follows from metadata auditing.
Formal manuscript work remains deferred; overall scientific acceptance is IN PROGRESS.
