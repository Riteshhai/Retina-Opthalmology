# QSART-X manuscript evidence tables

The field paths below refer to the linked aggregate JSON. `scenarios[id=…]`
selects a scenario by its stored `id`; comparisons use their stored candidate
and reference IDs. Matrix diagonal sums supply accuracy numerators where the
report stores a fraction and a confusion matrix. Retain the denominators and
study units when transferring values into a manuscript.

## Cohorts and roles

| Population | Images / eyes / patients | Role and evidence |
| --- | --- | --- |
| DeepDRiD Set-A fitting | 848 / 424 / 212 | Classifier fitting and paired-view value-model fitting. `fit_population`, `input_provenance.partitions.train` in the [paired-view report](../../reports/paired_view_v1/evaluation.json). |
| DeepDRiD Set-A selection | 168 / 84 / 42 | Previously inspected for classifier selection, and patient-disjoint from value-model fitting. `selection_population` in the [paired-view report](../../reports/paired_view_v1/evaluation.json). |
| DeepDRiD Set-A calibration | 168 / 84 / 42 | Calibration only. The Set-C sidecar used disjoint 21-patient temperature and conformal groups. `input_provenance.partitions.calibration` in the [paired-view report](../../reports/paired_view_v1/evaluation.json) and `input_provenance.calibration` in the [Set-C report](../../reports/set_c_validation_v1/evaluation.json). |
| DeepDRiD Set-B confirmation | 340 / — / 85 | Previously inspected historical internal confirmation. `images`, `patients` in the [Set-B report](../../reports/confirmation_v1/evaluation.json). |
| DeepDRiD Set-C source | 400 / 200 / 100 | One complete patient was excluded for a cross-split retinal fingerprint candidate. `population`, `overlap.reference_pairs`, and `overlap.reference_exact_pairs` in the [readiness report](../../reports/deepdrid_online_readiness_v1/evaluation.json). |
| DeepDRiD Set-C eligible | 396 / 198 / 99 | Same-release held-out evaluation. Two eyes and four photographs per eligible patient. `images`, `patients`, and `prediction_metrics.patient_grading.referable.positive_count` in the [Set-C report](../../reports/set_c_validation_v1/evaluation.json). |

The frozen production checkpoint IDs appear in `input_provenance.model_version`
and `input_provenance.quality_model_version` in both principal reports. The
Set-C report is bound to `eafd13db260bde7ed3627252d027e0cfc2ad2582` and the
paired-view report to `6017d0c889d539b9d8c74ab33907015388040acf`, the respective
pre-score source commits. [Set-C audit](../../reports/set_c_validation_v1/independent_audit.json) ·
[Paired-view audit](../../reports/paired_view_v1/independent_audit.json).

## Set-C predictions

| Measure | Value | Field in the [Set-C aggregate](../../reports/set_c_validation_v1/evaluation.json) |
| --- | ---: | --- |
| Five-grade photograph accuracy | 242/396 (61.1%) | `prediction_metrics.image_grading.{confusion_matrix,accuracy,n}` |
| Photograph macro-F1 and quadratic weighted kappa | 0.511 and 0.715 | `prediction_metrics.image_grading.{macro_f1,quadratic_weighted_kappa}` |
| Worst-predicted-view patient five-grade accuracy | 52/99 (52.5%) | `prediction_metrics.patient_grading.{worst_grade_confusion_matrix,worst_grade_accuracy,n}` |
| Patient grade-based referability | 45/48 positive patients detected, 15/51 negative patients flagged | `prediction_metrics.patient_grading.referable.{confusion_matrix,positive_count,n}` |
| AUROC for maximum view-level referable probability within a patient | 0.956 | `prediction_metrics.patient_grading.referable.auroc` |
| Image-quality accuracy at the fixed threshold | 299/396 (75.5%) | `prediction_metrics.quality.{accuracy,confusion_matrix,n,good_probability_threshold}` |
| Photo negative log likelihood, raw and frozen production calibration | 1.067 → 1.022 | `prediction_metrics.calibration.{raw,production}.nll` |
| Ordinal all-view inclusion and mean set size | 99/99 patients, 5.00 grades | `prediction_metrics.conformal.ordinal.{covered_patients,n_patients,mean_set_size,set_size_counts}` |
| LAC all-view inclusion and mean set size | 89/99 patients, 3.43 grades | `prediction_metrics.conformal.lac.{covered_patients,n_patients,mean_set_size}` |
| LAC grade-4 inclusion | 16/20 photographs | `prediction_metrics.conformal.lac.per_class["4"].{n,coverage}` |

The patient grade metrics use a maximum of predicted view grades for the
classification endpoint and a maximum of view probabilities for the ranking
endpoint. The high observed ranking discrimination coexisted with a policy
requirement to review all cases. [Set-C protocol](SET_C_VALIDATION_V1_PROTOCOL.md).

## Set-C policy and workload

| Deployed ordinal study arm | Image caps, review / recapture | Allocation | Field in the [Set-C aggregate](../../reports/set_c_validation_v1/evaluation.json) |
| --- | ---: | --- | --- |
| 30% / 20% | 118 / 79 | Overload. Review shortfall 278. | `scenarios[id=ordinal:dual_budget:0].{status,diagnostics}` |
| 60% / 40% | 237 / 158 | Overload. Review shortfall 159. | `scenarios[id=ordinal:dual_budget:1].{status,diagnostics}` |
| 90% / 40% | 356 / 158 | Overload. Review shortfall 40. | `scenarios[id=ordinal:dual_budget:2].{status,diagnostics}` |
| 100% / 40% | 396 / 158 | All 396 photographs and 99 patients assigned to review. Conditional automatic clearance miss fraction undefined. | `scenarios[id=ordinal:dual_budget:3].{budget,outcomes.image_actions,outcomes.patient_actions,outcomes.full_patient_clearance_miss_fraction}` |

At the 60% / 40% image budget without conformal, the photo action counts were
45 automatic, 114 recapture, and 237 review. There were four referable and one
severe photo misses within the automatic group. Patient action counts were
zero fully automatic, 25 recapture, and 74 review. The patient diagnostic's
review cap was 59, with at least 71 slots required by pointwise recommendations.
See `scenarios[id=no_cp:dual_budget:1].outcomes`, including
`missed_referable_images` and `missed_severe_images`, and the scenario's
`patient_capacity.{review_capacity,minimum_review_after_spill}`. The LAC
diagnostic has the same overload pattern as the ordinal arm for dual-budget
allocation, and a full-review recommendation at `scenarios[id=lac:dual_budget:3]`.
[Scenario records](../../reports/set_c_validation_v1/evaluation.json).

| Study-wide result | Value | Trace |
| --- | ---: | --- |
| Allocations evaluated or overloaded | 39 feasible and 57 overloaded out of 96 | `evaluated_allocations`, `overloaded_allocations`, and `scenarios_verified` in the [independent audit](../../reports/set_c_validation_v1/independent_audit.json). |
| Maximum fully automatic patients in a feasible scenario | 1 of 99 | Maximum `outcomes.patient_actions.AUTO_NONREFERABLE` among `scenarios` with `status=evaluated` in the [Set-C report](../../reports/set_c_validation_v1/evaluation.json). |
| Workload-matched policy pairs | 12 exact, 22 unequal realized workloads, 50 unavailable out of 84 | `comparisons_verified`, `exact_workload_matched_pairs`, and `unavailable_comparisons` in the [independent audit](../../reports/set_c_validation_v1/independent_audit.json). The unequal count is `84 − 12 − 50`. |
| Identical image recommendations in exactly matched pairs | 12/12 | `comparisons` filtered where `matched_image_workload=true` and `matched_patient_workload=true`; all have `identical_image_actions=true` in the [Set-C report](../../reports/set_c_validation_v1/evaluation.json). |

**Figure caption.** Patient action rollups under each Set-C dual-budget image
capacity configuration. Overloads lack a feasible allocation. The bars show
recommended actions for 99 patients, calculated from the four photo actions
per patient. [Patient dispositions figure](../../reports/set_c_validation_v1/patient_dispositions.png).

## Paired-view acquisition

| Selected scenario | Extra fields | Eye missed referrals, all / severe | Patient missed referrals, all / severe | Review proxy | Ledger per eye | Scenario `id` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| First view only | 0 | 8 / 4 | 3 / 1 | 36 | 0.5833 | `first_view:0` |
| Value rank, 10% cap | 8 | 6 / 3 | 2 / 1 | 38 | 0.4536 | `voi:1` |
| Value rank, 25% cap | 21 | 6 / 3 | 2 / 1 | 39 | 0.4875 | `voi:2` |
| Value rank, 50% cap | 42 | 6 / 3 | 2 / 1 | 39 | 0.5375 | `voi:3` |
| Value rank, 75% cap | 63 | 6 / 3 | 2 / 1 | 39 | 0.5875 | `voi:4` |
| Both recorded fields | 84 | 6 / 3 | 2 / 1 | 36 | 0.6286 | `both_views:5` |
| Acquire above estimated cost, any interior cap | 4 | 8 / 4 | 3 / 1 | 36 | 0.5929 | `voi_positive:1` through `voi_positive:4` |

The data come from `scenarios[id=…].outcomes.{extra_views,referable_misses,
severe_misses,patient_referable_misses,patient_severe_misses,review_proxy_count,
ledger_per_eye}`. The comparator evidence comprises 84 eyes from 42 selection
patients, including 37 referable and 25 severe eyes. The number of views used
by the capped rule may be smaller than the available `scenarios[id=…].slots`.
[Paired-view aggregate](../../reports/paired_view_v1/evaluation.json).

For the eight-field comparison against entropy, select the comparison whose
`candidate=voi:1` and `reference=entropy:1`. Its
`intervals.ledger_per_eye.{difference_candidate_minus_reference,ci95}` gives
−0.1161 [−0.3530, +0.0095]. All 16 entries in `comparisons` contain zero in
`intervals.ledger_per_eye.ci95`. At 21 and 42 acquisitions, the positive
value-minus-entropy estimates favor entropy on the observed ledger. The audit
reconstructed all 26 scenarios and 16 paired intervals from saved scores.
[Exact-workload comparisons](../../reports/paired_view_v1/evaluation.json) ·
[Independent audit](../../reports/paired_view_v1/independent_audit.json).

**Figure caption.** Mean research ledger per selection eye versus the number
of additional recorded fields used. The first-view and both-field arms anchor
the fixed-budget comparison. The cost-threshold rule uses four fields at all
interior budgets. The study measures a retrospective workload and error proxy.
[Acquisition workload figure](../../reports/paired_view_v1/acquisition_workload.png).

## Separate context studies

| Study and evaluation unit | Selected observation | Source field |
| --- | --- | --- |
| DeepDRiD development horizontal flip, photograph | At 42 extra forwards, entropy correctly classified 109/168 grades versus 106/168 at first pass. Extra passes represent transformations of existing photographs. | `images`, `anchors.single.grade.correct`, `budgets[budget=.25].rankings.entropy.metrics.grade.correct` in the [flip report](../../reports/matched_flip_v1/evaluation.json). |
| DeepDRiD development controlled degradation, photograph | Clean accuracy 63.10%, JPEG quality 20 accuracy 22.02%. | `conditions.{clean,jpeg_20}.grading.accuracy` in the [degradation report](../../reports/robustness_v1/evaluation.json). |
| FDR26 v3, photograph | The frozen DeepDRiD classifier matched 794 of 2,849 released ALFEDIAM codes (27.87%). Patient linkage was unavailable. | `populations.primary.{images,calibrated.accuracy,calibrated.confusion}`, `patient_linkage_available` in the [FDR26 report](../../reports/fdr26_grading_v1/evaluation.json). |
| IDRiD, fingerprint representative photograph | The prespecified APTOS fine-tuned seed-2026 model achieved 42.97% grade accuracy on 505 representatives. Prior outcomes were inspected and patient linkage was unavailable. | `models[model_id=finetune_seed2026].grading.cross_source_filtered_representatives.{n,accuracy}`, `external_outcomes_consumed`, and `population.patient_identity_available` in the [IDRiD report](../../reports/idrid_external_v1/evaluation.json). |

These studies use different annotation standards, tested models, populations,
or outcomes. Their aggregate measures do not define a pooled patient screening
estimate. [FDR26 source deposit](https://zenodo.org/records/23101200) ·
[IDRiD primary descriptor](https://doi.org/10.3390/data3030025).

## IEEE table layout

The IEEEtran export contains both primary tables and references to both
checked-in figures. Run `python scripts/export_qsart_x_tables.py --check` to
verify it against the audit-bound aggregates. [LaTeX tables and figures](QSART_X_TABLES.tex).

## Reproduction checklist

1. Use the dataset guide to recover the permitted original images and bound
   manifests, checkpoint files, and calibration artifacts. Keep local rows
   and private witnesses outside version control. [Dataset guide](../QSART_X_DATASETS.md).
2. Check out each report's declared pre-score source commit in a clean checkout
   with empty output paths. Preserve input bytes, including the development
   summary's original line endings, to match the report's hashes. Use the
   recorded run, audit, and figure commands for each experiment.
   [Set-C reproduction](SET_C_VALIDATION_V1_RESULTS.md#audit-and-reproduction) ·
   [Paired-view reproduction](PAIRED_VIEW_V1_RESULTS.md#audit-and-reproduction).
3. Verify `audit_status=passed` and the receipt's `evaluation_sha256` against
   the aggregate bytes before updating a manuscript number. Preserve separate
   counts for infeasible, unequal-workload, and matched arms.
   [Set-C receipt](../../reports/set_c_validation_v1/independent_audit.json) ·
   [Paired-view receipt](../../reports/paired_view_v1/independent_audit.json).
