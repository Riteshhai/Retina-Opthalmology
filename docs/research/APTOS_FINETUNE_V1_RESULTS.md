# APTOS parameter fine-tuning v1 — results

**VERIFIED COMPLETE:** the bounded three-seed adaptation pilot and independent
artifact/mathematical audit. Overall scientific acceptance remains IN PROGRESS.
The [protocol](APTOS_FINETUNE_V1_PROTOCOL.md) declares all decisions before real
fine-tuning. Execution source is `8357e9d918ef9ec00d1f66e4efadc3325790097c`; measured report SHA256
is `8c6f50e93db412fc845600a4e368094c46fce041109806e37819dcd6bb70c5f4`. Primary seed 2026 was fixed before outcomes, and no checkpoint,
seed, alpha or uncertainty method is promoted to the application.

## Design and cohort

All three runs share the previously selected EfficientNet-B0 checkpoint and
five-way APTOS head. Every learnable backbone/head parameter adapts, while all
BatchNorm running buffers stay fixed. The model retains stochastic depth during
training. Each run uses four epochs maximum, batch 16, AdamW with initial backbone
LR .0001/head .0003, weight decay .0001, cosine T_max 4 / minimum LR .000001, class-balanced
cross entropy and random horizontal flips. Four CPU threads and channels-last
memory layout were chosen in a pre-execution software capacity check. This short
warm-start pilot neither establishes convergence nor reproduces the paper's
384px full optimization protocol.

The original 148-image/51-conflicting-component quarantine and fixed partitions
are preserved: 2,115 fitting, 521 selection, 527 calibration and 351 reserved
internal-test images. The active relocated source is mapped privately and every
development byte is verified. No reserved-test model inference occurs. Initial
source labels had already been read for quarantine/stratification. Fingerprint
components are not patients; unknown related views and label-dependent exclusion
change the population and limit independence assumptions.

One-pass deterministic RGB geometry is cached as uint8. Training augmentation
occurs after cache reading, and RNGs are reseeded after initialization/cache work.
The cache-hit/miss and exact interrupted/resumed training regressions pass.
The real pre-optimizer check matched all 521 original predicted grades, with
maximum probability delta 6e-06
under the declared 1e-5 tolerance. All selected/last checkpoints show changed
backbone parameters and unchanged BatchNorm buffers.

## Classification outcomes

These are the SAME previously inspected selection images used to choose
checkpoints. The table reports uncalibrated predictions; temperature preserves
the argmax. The warm start's selected epoch belongs to the historical head-only
training procedure and is not comparable training duration.

| Run | Selected epoch | Accuracy | Macro-F1 | QWK | Grade-3 recall (n=26) | Grade-4 recall (n=40) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen-feature warm start | 21 | 78.69% | 0.60334 | 0.84400 | 23.08% | 55.00% |
| Fine-tuning seed 2026 | 4 | 78.31% | 0.63349 | 0.86005 | 65.38% | 47.50% |
| Fine-tuning seed 2027 | 4 | 78.50% | 0.62078 | 0.85665 | 30.77% | 70.00% |
| Fine-tuning seed 2028 | 4 | 77.54% | 0.61807 | 0.85669 | 38.46% | 55.00% |

Across-training-seed means and SAMPLE standard deviations, conditional on the
shared warm start and split:

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| accuracy | 0.781190 | 0.005078 |
| macro_f1 | 0.624113 | 0.008232 |
| quadratic_weighted_kappa | 0.857795 | 0.001955 |
| nll | 0.677878 | 0.039785 |
| ece | 0.084904 | 0.010193 |
| brier | 0.318866 | 0.014887 |
| grade_0_recall | 0.951654 | 0.017629 |
| grade_1_recall | 0.679739 | 0.081634 |
| grade_2_recall | 0.622066 | 0.026662 |
| grade_3_recall | 0.448718 | 0.181762 |
| grade_4_recall | 0.575000 | 0.114564 |

All runs and all per-grade supports/confusion matrices remain in the aggregate
report. Inspect grade-specific tradeoffs alongside macro-F1 and accuracy; a gain
in one grade is not uniform severity improvement. Three seeds do not estimate
full-pipeline uncertainty, and no favorable seed is selected.

All three selected models have higher macro-F1 and lower accuracy than the warm
start on this selection population. Primary seed 2026 raises grade-3 recall from
6/26 to 17/26, while grade-4 recall falls from 22/40 to 19/40. The other seeds show
different severity tradeoffs. Every paired accuracy and macro-F1 interval below
includes zero; the study does not establish superiority.

## Paired changes and calibration

Each comparison pairs the frozen baseline and fine-tuned predictions on the same
521 images, with 1,000 fingerprint-component resamples, seed 20261104. Whole groups
and duplicate draws are retained. The descriptive 95% percentile intervals
condition on fixed selected predictions; there is no refitting, multiplicity
adjustment, patient-independence guarantee or fresh-test superiority claim.

| Training seed | Accuracy delta [lower,upper] | Macro-F1 delta [lower,upper] | Temperature | Raw NLL -> calibrated NLL | Raw ECE10 -> calibrated ECE10 |
| --- | --- | --- | ---: | --- | --- |
| 2026 | -0.00384 [-0.03288,+0.02662] | +0.03015 [-0.02346,+0.08355] | 1.735318 | 0.703504 -> 0.591863 | 0.091997 -> 0.039433 |
| 2027 | -0.00192 [-0.03019,+0.02713] | +0.01744 [-0.03225,+0.07137] | 1.777133 | 0.632045 -> 0.542345 | 0.089491 -> 0.050326 |
| 2028 | -0.01152 [-0.04398,+0.01927] | +0.01473 [-0.03539,+0.06374] | 1.636886 | 0.698086 -> 0.616755 | 0.073223 -> 0.027317 |

Every model fits temperature on the same independent-role 262 images; the other
role supplies 254 conformal representatives. Calibration roles remain separate
from classifier fitting and checkpoint selection. These previously inspected
calibration data are reused transparently. ECE here uses 10 bins; separate
temperature-fit diagnostics use 15 bins and must not be mixed in before/after
comparisons. Raw and calibrated across-seed summaries are retained separately.

## All uncertainty comparisons

All 42 declared rows are retained. Training seeds identify classifiers; the five
randomization seeds identify explicit uniform draws. Each row below reports all
521 selection images. Raw empties and the common argmax enlargement are distinct.
Full reports additionally retain 507 selection representatives, every grade's
coverage/support, contiguity, all-five counts and size distributions.

| Model seed | Alpha | Method | Uniform seed | Raw coverage / mean size / empties | Fallback coverage / mean size | Grade-3 / grade-4 fallback coverage |
| ---: | ---: | --- | --- | --- | --- | --- |
| 2026 | 0.10 | lac | none | 90.79% / 1.445 / 0 | 90.79% / 1.445 | 80.77% / 75.00% |
| 2026 | 0.10 | ordinal_greedy | none | 89.44% / 2.656 / 44 | 97.89% / 2.741 | 100.00% / 82.50% |
| 2026 | 0.10 | ordinal_greedy_randomized | 2026 | 92.71% / 1.864 / 9 | 94.43% / 1.881 | 96.15% / 72.50% |
| 2026 | 0.10 | ordinal_greedy_randomized | 2027 | 89.64% / 1.695 / 22 | 93.86% / 1.737 | 92.31% / 75.00% |
| 2026 | 0.10 | ordinal_greedy_randomized | 2028 | 89.83% / 1.679 / 19 | 93.47% / 1.716 | 92.31% / 72.50% |
| 2026 | 0.10 | ordinal_greedy_randomized | 2029 | 90.79% / 1.720 / 16 | 93.86% / 1.750 | 92.31% / 72.50% |
| 2026 | 0.10 | ordinal_greedy_randomized | 2030 | 92.13% / 1.783 / 11 | 94.24% / 1.804 | 96.15% / 72.50% |
| 2026 | 0.05 | lac | none | 94.63% / 1.777 / 0 | 94.63% / 1.777 | 84.62% / 85.00% |
| 2026 | 0.05 | ordinal_greedy | none | 97.12% / 3.305 / 9 | 98.85% / 3.322 | 100.00% / 90.00% |
| 2026 | 0.05 | ordinal_greedy_randomized | 2026 | 95.97% / 2.207 / 5 | 96.93% / 2.217 | 100.00% / 80.00% |
| 2026 | 0.05 | ordinal_greedy_randomized | 2027 | 95.39% / 2.017 / 6 | 96.55% / 2.029 | 100.00% / 80.00% |
| 2026 | 0.05 | ordinal_greedy_randomized | 2028 | 94.82% / 2.044 / 9 | 96.55% / 2.061 | 92.31% / 85.00% |
| 2026 | 0.05 | ordinal_greedy_randomized | 2029 | 95.01% / 2.044 / 7 | 96.35% / 2.058 | 96.15% / 77.50% |
| 2026 | 0.05 | ordinal_greedy_randomized | 2030 | 95.20% / 2.046 / 6 | 96.35% / 2.058 | 100.00% / 80.00% |
| 2027 | 0.10 | lac | none | 90.21% / 1.403 / 0 | 90.21% / 1.403 | 61.54% / 87.50% |
| 2027 | 0.10 | ordinal_greedy | none | 92.90% / 3.067 / 29 | 98.46% / 3.123 | 100.00% / 90.00% |
| 2027 | 0.10 | ordinal_greedy_randomized | 2026 | 94.24% / 1.973 / 10 | 96.16% / 1.992 | 96.15% / 85.00% |
| 2027 | 0.10 | ordinal_greedy_randomized | 2027 | 90.98% / 1.820 / 22 | 95.01% / 1.862 | 96.15% / 85.00% |
| 2027 | 0.10 | ordinal_greedy_randomized | 2028 | 90.60% / 1.831 / 20 | 94.43% / 1.869 | 92.31% / 87.50% |
| 2027 | 0.10 | ordinal_greedy_randomized | 2029 | 91.75% / 1.831 / 17 | 95.01% / 1.864 | 92.31% / 82.50% |
| 2027 | 0.10 | ordinal_greedy_randomized | 2030 | 91.36% / 1.854 / 15 | 94.24% / 1.883 | 96.15% / 82.50% |
| 2027 | 0.05 | lac | none | 95.78% / 1.752 / 0 | 95.78% / 1.752 | 80.77% / 90.00% |
| 2027 | 0.05 | ordinal_greedy | none | 95.97% / 3.512 / 13 | 98.46% / 3.537 | 100.00% / 90.00% |
| 2027 | 0.05 | ordinal_greedy_randomized | 2026 | 96.74% / 2.369 / 5 | 97.70% / 2.378 | 96.15% / 87.50% |
| 2027 | 0.05 | ordinal_greedy_randomized | 2027 | 95.78% / 2.165 / 6 | 96.93% / 2.177 | 96.15% / 87.50% |
| 2027 | 0.05 | ordinal_greedy_randomized | 2028 | 95.59% / 2.184 / 9 | 97.31% / 2.202 | 96.15% / 90.00% |
| 2027 | 0.05 | ordinal_greedy_randomized | 2029 | 95.39% / 2.148 / 8 | 96.93% / 2.163 | 96.15% / 82.50% |
| 2027 | 0.05 | ordinal_greedy_randomized | 2030 | 96.55% / 2.230 / 5 | 97.50% / 2.240 | 96.15% / 90.00% |
| 2028 | 0.10 | lac | none | 90.21% / 1.545 / 0 | 90.21% / 1.545 | 76.92% / 77.50% |
| 2028 | 0.10 | ordinal_greedy | none | 91.94% / 2.831 / 30 | 97.70% / 2.889 | 100.00% / 85.00% |
| 2028 | 0.10 | ordinal_greedy_randomized | 2026 | 91.94% / 1.914 / 8 | 93.47% / 1.929 | 88.46% / 75.00% |
| 2028 | 0.10 | ordinal_greedy_randomized | 2027 | 90.21% / 1.827 / 15 | 93.09% / 1.856 | 88.46% / 80.00% |
| 2028 | 0.10 | ordinal_greedy_randomized | 2028 | 88.29% / 1.760 / 20 | 92.13% / 1.798 | 80.77% / 80.00% |
| 2028 | 0.10 | ordinal_greedy_randomized | 2029 | 90.40% / 1.827 / 12 | 92.71% / 1.850 | 88.46% / 75.00% |
| 2028 | 0.10 | ordinal_greedy_randomized | 2030 | 91.55% / 1.841 / 10 | 93.47% / 1.860 | 88.46% / 77.50% |
| 2028 | 0.05 | lac | none | 95.39% / 1.898 / 0 | 95.39% / 1.898 | 80.77% / 90.00% |
| 2028 | 0.05 | ordinal_greedy | none | 96.35% / 3.353 / 11 | 98.46% / 3.374 | 100.00% / 90.00% |
| 2028 | 0.05 | ordinal_greedy_randomized | 2026 | 96.16% / 2.282 / 3 | 96.74% / 2.288 | 92.31% / 85.00% |
| 2028 | 0.05 | ordinal_greedy_randomized | 2027 | 94.24% / 2.069 / 4 | 95.01% / 2.077 | 92.31% / 85.00% |
| 2028 | 0.05 | ordinal_greedy_randomized | 2028 | 94.63% / 2.136 / 7 | 95.97% / 2.150 | 92.31% / 90.00% |
| 2028 | 0.05 | ordinal_greedy_randomized | 2029 | 94.24% / 2.125 / 6 | 95.39% / 2.136 | 92.31% / 80.00% |
| 2028 | 0.05 | ordinal_greedy_randomized | 2030 | 96.35% / 2.286 / 3 | 96.93% / 2.292 | 96.15% / 87.50% |

For primary seed 2026 at alpha .10, LAC marginal inclusion is 90.79%, but grade-4
inclusion is only 75.00%. All five randomized ordinal fallback replicates include
only 72.50–75.00% of grade-4 images. Higher aggregate inclusion does not remove
the severe-grade failure.

Aggregate inclusion can hide severe-grade exclusions. None of these development
diagnostics establishes patient, grade-conditional, external or clinical coverage.
An argmax fallback enlarges raw sets and must not be described as the raw method.
No seed or threshold was retuned to improve these displayed outcomes.

## Artifacts and limitations

The [full report](../../reports/aptos_finetune_v1/evaluation.json),
[independent audit](../../reports/aptos_finetune_v1/independent_audit.json),
[grading figure](../../reports/aptos_finetune_v1/grading_by_seed.png) and
[uncertainty figure](../../reports/aptos_finetune_v1/conformal_diagnostics.png)
provide exact hashes, histories, counts, input/cache bindings and package versions.
Private images, manifests, checkpoints, epoch logits, calibrated scores and
individual sets remain local and ignored. Original APTOS/DeepDRiD results,
production checkpoints/sidecars and master specification bytes are preserved.

The audit reconstructs every epoch metric/checkpoint choice, schedule and flip
counts, selected/last parameter/BatchNorm states, three temperatures, six paired
intervals and 42 raw/fallback comparisons (43,764 individual sets). Its recorded
scope and limitations are authoritative. Native API and Streamlit component
verification are separate from new container or browser-rendering evidence.

After the scientific audit, the plotter's x-axis label was shortened to prevent
overlap. The measured aggregate stayed byte-identical. Verification records this
presentation change separately from the pinned training execution source.

Remaining requirements include stronger/converged optimization, independent
reserved-test/external evaluation, authoritative quality supervision and useful
validated triage. Formal manuscript drafting remains deferred.
