# DeepDRiD Set-C patient validation

The deployed ordinal policy required specialist review for all 396 photos.
It overloaded at 30%, 60% and 90% image review capacity. At 100% capacity
it recommended review for all 99 patients. With zero fully automatic patients,
the conditional miss rate was undefined. [Locked protocol](SET_C_VALIDATION_V1_PROTOCOL.md).

The frozen models scored two photos per eye from 99 eligible patients in the
same-release final online split. The readiness audit excluded one complete
patient for a cross-split fingerprint match. The released worse-eye grades
identified 48 referable and 28 severe patients under the study definitions.
[Readiness evidence](DEEPDRID_ONLINE_READINESS_V1_RESULTS.md) ·
[DeepDRiD data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/).

## Fixed dual-budget policy

| Gate | Image review / recapture cap | Photo actions (automatic / recapture / review) | Patient actions (automatic / recapture / review) |
| --- | ---: | ---: | ---: |
| Deployed ordinal | 30% / 20% | Overload | — |
| Deployed ordinal | 60% / 40% | Overload | — |
| Deployed ordinal | 90% / 40% | Overload | — |
| Deployed ordinal | 100% / 40% | 0 / 0 / 396 | 0 / 0 / 99 |
| LAC diagnostic | 30% / 20% | Overload | — |
| LAC diagnostic | 60% / 40% | Overload | — |
| LAC diagnostic | 90% / 40% | Overload | — |
| LAC diagnostic | 100% / 40% | 0 / 0 / 396 | 0 / 0 / 99 |
| No-conformal diagnostic | 30% / 20% | Overload | — |
| No-conformal diagnostic | 60% / 40% | 45 / 114 / 237 | 0 / 25 / 74 |
| No-conformal diagnostic | 90% / 40% | 4 / 36 / 356 | 0 / 0 / 99 |
| No-conformal diagnostic | 100% / 40% | 0 / 0 / 396 | 0 / 0 / 99 |

Without conformal at the 60% / 40% image caps, four referable photographs
received automatic recommendations, including one severe photograph. No patient
was fully cleared, and one severe patient received a recapture recommendation.
That allocation fit both image caps. Its summarized review count was 74 patients
against 59 patient review slots. The separate pointwise patient-capacity check
needed at least 71 review slots. The capacity units matter for workload planning.

Across all eight guarded methods and three modes, 39 of 96 allocations fit
both image caps and 57 overloaded. No evaluated allocation fully cleared more
than one of 99 patients. All instances of a fully automatic patient arose in
the no-conformal diagnostic, with no referable patient among those cleared.
The largest conditional clearance denominator was one. Review and recapture
outcomes were unobserved.

## Matched workloads

Of 84 comparisons, 50 had an overload in at least one arm and 22 differed in
realized workloads. The remaining 12 matched both review and recapture counts
at both image and patient level. All 12 made identical image decisions. Their
paired patient differences and descriptive intervals were zero. The comparison
supplies no evidence that optimizing the proposed objective improved decisions
at the same realized workload on this split.

## Prediction and uncertainty

| Metric | Measured result |
| --- | ---: |
| Five-grade photo accuracy | 242 / 396 (61.1%) |
| Photo macro-F1 / quadratic weighted kappa | 0.511 / 0.715 |
| Five-grade patient accuracy using worst predicted view | 52 / 99 (52.5%) |
| Patient referability from worst predicted view | 45 of 48 detected, 15 of 51 false positives |
| Patient referable ranking AUROC (maximum photo probability) | 0.956 |
| Image-quality accuracy at the fixed threshold | 299 / 396 (75.5%) |
| Ordinal patient coverage / mean set size | 99 / 99, 5.00 grades |
| LAC patient coverage / mean set size | 89 / 99, 3.43 grades |

The ordinal gate returned all five grades for every photograph. Its coverage
was therefore automatic by set construction. LAC covered 381 of 396 photographs
and all four views for 89 patients, with 16 of 20 grade-4 photographs covered.
Frozen production temperature reduced photo log loss from 1.067 to 1.022. The
study reports empirical performance for the released labels and keeps those
probability diagnostics separate from screening outcomes.

The previously audited development stress test and external results retain
their original scopes. JPEG quality 20 reduced accuracy to 22.02% on the
development selection cohort. On the FDR26 primary grading population, the
frozen DeepDRiD model agreed with 794 of 2,849 released ALFEDIAM codes. IDRiD
evaluation remains previously inspected external evidence. These studies use
different populations or perturbations and are presented separately.
[Controlled degradation](ROBUSTNESS_V1_RESULTS.md) ·
[FDR26 grading](FDR26_GRADING_V1_RESULTS.md) · [IDRiD](IDRID_EXTERNAL_V1_RESULTS.md).

## Audit and reproduction

The independent auditor bound the report to the locked source and private
witness. It reconstructed the complete eligible roster, both conformal set
methods from saved raw scores, the 96 capacity and allocation checks, and all
84 comparison records. The audit used saved scores for statistical and
allocation reconstruction. Set-C outcomes have now been inspected and must
stay outside fitting, threshold tuning and later claims of fresh confirmation.

Run from clean study commit `eafd13db260bde7ed3627252d027e0cfc2ad2582` with
the bound readiness witness, model files and authorized source package.
Use a checkout with empty study-output paths and supply the local author
package path to `--root`.

```text
python -u scripts/run_set_c_validation.py --root PATH_TO_AUTHOR_PACKAGE --source-commit eafd13db260bde7ed3627252d027e0cfc2ad2582
python scripts/audit_set_c_validation.py
python scripts/plot_set_c_validation.py
```

[Aggregate report](../../reports/set_c_validation_v1/evaluation.json) ·
[Audit receipt](../../reports/set_c_validation_v1/independent_audit.json) ·
[Patient workload figure](../../reports/set_c_validation_v1/patient_dispositions.png).
