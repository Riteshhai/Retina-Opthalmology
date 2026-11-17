# Fixed APS/RAPS comparison v1 — 6 November 2026

**VERIFIED COMPLETE:** fixed saved-prediction score comparison, independent audit
and native/container integration. Overall scientific acceptance remains IN PROGRESS.
Execution source `e7fd2f1b85d79e2f9f910433eea0dd7beea06b7c` was committed before new
score outcomes. [Declared protocol](ADAPTIVE_SETS_V1_PROTOCOL.md).

The primary fixed RAPS setting produces smaller sets than APS and lower severe-
grade inclusion. This tradeoff does not support a production replacement. Every
unfavorable outcome, raw empty set, fallback result, uniform seed and interval is retained.

## Scope and source boundary

One existing APTOS EfficientNet-B0, training seed 2026/selected epoch 10, retains its
temperature 2.940995751533916. The new study fits 34 score thresholds from saved
logits, with no image decoding, neural inference, classifier/temperature fitting
or RAPS parameter selection. The deployment verification separately performs real
DeepDRiD inference with the unchanged production model. Neither activity evaluates
reserved APTOS tests or produces new EyeQ/IDRiD outcomes.

The 254 conformal representatives remain separate from 262 temperature-role images.
Both alpha levels use all 521 previously inspected selection images and 507 observed
fingerprint representatives. Earlier calibration/selection and IDRiD outcome
inspection is disclosed. Component grouping, grade-stratified partitions and
label-dependent quarantine do not establish patient independence or exchangeability.
This is reused development evidence, with no fresh confirmation or coverage guarantee.

APS uses ranked preceding mass plus (1-u) times candidate probability. Fixed RAPS
adds .1 max(rank-2,0); full-mass/randomized ordinal and LAC remain distinct controls.
The [primary RAPS paper](https://arxiv.org/html/2009.14193v5) supplies the score
construction. The pinned [retinopathy author code](https://github.com/umarbhasan/conformal-diabetic-retinopathy/blob/3a7d18d799262098d8166519159d538960bbf118/src.py)
supplies the initial CFG setting k_reg=2/lambda=.10; its later tuned procedure is
not reproduced. QSART independently implements stable rank ties, explicit common
uniforms, exact finite-sample ranks and raw/fallback reporting. Author code is not
imported or executed. Full 384px/backbone/five-training-seed/tuned-paper reproduction
remains incomplete. No parameter, method, seed or new sidecar is selected for production.

## Primary comparison

Alpha .10, uniform seed 2026, raw sets, all 521 selection rows. Grade 3 support 26;
grade 4 support 40. Inclusion measures whether the source label is in the set.

| Method | Inclusion | Mean size | Empty sets | Grade 3 inclusion | Grade 4 inclusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAC | 89.25% | 1.29559 | 0 | 46.15% | 70.00% |
| APS | 90.98% | 1.63916 | 15 | 69.23% | 80.00% |
| Fixed RAPS | 90.40% | 1.53935 | 12 | 53.85% | 80.00% |
| Randomized ordinal | 91.36% | 1.76008 | 16 | 96.15% | 75.00% |
| Full-mass ordinal | 90.21% | 2.58349 | 42 | 100.00% | 82.50% |

RAPS reduces mean set size 1.63916 to 1.53935, while grade 3 inclusion falls 18/26 to
14/26 and overall inclusion falls 90.98% to 90.40%. Randomized ordinal gives wider
sets than APS and grade 3 inclusion 25/26, but grade 4 inclusion is 30/40 versus APS 32/40.
Wider/contiguous sets and near-target marginal inclusion do not establish severe-
grade or clinical adequacy. Fingerprint-representative sensitivity results are
retained separately, with grade 3 support 24 and grade 4 support 39.

Argmax fallback enlarges only raw empty sets and changes the score-event equality:

| Method | Fallback inclusion | Fallback mean size |
| --- | ---: | ---: |
| LAC | 89.25% | 1.29559 |
| APS | 93.86% | 1.66795 |
| Fixed RAPS | 92.71% | 1.56238 |
| Randomized ordinal | 94.43% | 1.79079 |
| Full-mass ordinal | 98.27% | 2.66411 |

## Paired descriptive changes

These fixed-model/threshold/uniform intervals retain complete fingerprint
components, including repeated draws. All 2,000 draws use bootstrap seed 20261109.
There is no refitting, reselection, multiplicity adjustment or patient-identity
claim. The primary raw contrasts below are a subset of all 160 retained intervals.

| Direction | Endpoint | Change | Descriptive 95% interval |
| --- | --- | ---: | --- |
| RAPS minus APS | mean_set_size | -0.09981 | [-0.13206, -0.06897] |
| RAPS minus APS | marginal_coverage | -0.00576 | [-0.02076, +0.00765] |
| RAPS minus APS | grade_3_inclusion | -0.15385 | [-0.30435, -0.03448] |
| RAPS minus APS | grade_4_inclusion | +0.00000 | [+0.00000, +0.00000] |
| Randomized ordinal minus APS | mean_set_size | +0.12092 | [+0.08670, +0.15667] |
| Randomized ordinal minus APS | marginal_coverage | +0.00384 | [-0.00965, +0.01894] |
| Randomized ordinal minus APS | grade_3_inclusion | +0.26923 | [+0.11111, +0.43478] |
| Randomized ordinal minus APS | grade_4_inclusion | -0.05000 | [-0.12821, +0.00000] |

All 34 comparisons, both empty policies and both populations yield 136 aggregate
metric views and 35,428 privately retained per-image set rows. Five uniform seeds
are randomization replicates of one fixed trained model; their variation is not
training uncertainty. All five seeds, alpha .05, every grade support/inclusion,
threshold/rank, empty/full-set count and contiguity/size distribution remain in
the report and explorer. Twenty conditional mean/sample-SD rows are independently
checked without selecting a favorable replicate.

## Independent verification and running application

The independent loop/count verifier imports neither new scoring nor study module.
It reconstructs all 34 score thresholds (maximum absolute discrepancy 1.110223e-16),
all 136 aggregate metric views, 35,428 raw/fallback set rows and 160 intervals. All 14
historical LAC/ordinal sets and stored uniforms reproduce exactly. A separate check
reconstructs 20 conditional summaries / 120 mean-SD cells. All 1,048 saved input-logit
rows are hash-verified, and all 295 historical/master/production/source artifacts
remain unchanged. Images, identifiers, raw logits, sets and weights remain local.

Before execution, all 593 tests passed in 109.14s, zero failures/errors/skips, with
genuine production/conformal integration; 35 targeted tests passed in 12.89s. One
upstream AnyIO deprecation warning remains. Tested scientific/API/UI source stayed
fixed. Ruff, Compose and whitespace checks pass. The exact
[source CI](https://github.com/Riteshhai/Retina-Opthalmology/actions/runs/42086591713)
passed. The saved-result figure was visually inspected.

Actual native and rebuilt nonroot/read-only CPU-container services pass genuine
JPEG single/batch/TTA, readiness, matched model card/sidecar, simulation and overload
checks. Each live-API AppTest verifies 20 evidence records, 40 control combinations,
2,200 metric cells and 1,280 interval cells. Default alpha/seed/raw/all-image controls
and explicit reuse/randomization/grouping captions match the report. All 68
container source/configuration files match local bytes. AppTest uses host Streamlit
against actual HTTP; no new interactive browser-rendering proof is claimed.
Temporary verification containers on 28000/28501 were removed after checks. The
native app remains on 18000/18501, with production artifacts unchanged.

## Files and next work

Aggregate SHA256:
`a883d166fe7172d77ee615b300ae46e802355c19731ecf9080640c3b41fc0e88`.

- [Aggregate](../../reports/adaptive_sets_v1/evaluation.json), [independent audit](../../reports/adaptive_sets_v1/independent_audit.json) and [summary check](../../reports/adaptive_sets_v1/summary_audit.json)
- [Combined verification](../../reports/adaptive_sets_v1/verification.json)
- [Native deployment](../../reports/adaptive_sets_v1/native_deployment.json) and [explorer](../../reports/adaptive_sets_v1/native_registry.json)
- [Container runtime](../../reports/adaptive_sets_v1/container_runtime.json), [deployment](../../reports/adaptive_sets_v1/container_deployment.json) and [explorer](../../reports/adaptive_sets_v1/container_registry.json)
- [Primary figure](../../reports/adaptive_sets_v1/primary_set_comparison.png) and [SVG](../../reports/adaptive_sets_v1/primary_set_comparison.svg)

The local EyeQ-master copy matches all seven pinned metadata/source files but
contains only five example JPEGs, not the full photograph cohort. No example was
decoded or promoted to independent validation. Official EyeQ photograph/terms/
storage and cross-source overlap/harmonization readiness remain incomplete.

Next feasible source work is a separately locked 384px training/compute feasibility
protocol for closer paper replication, preserving current studies and reserved
tests. Reliable quality supervision, useful validated triage and broader independent
validation remain open. Formal manuscript drafting stays deferred.
