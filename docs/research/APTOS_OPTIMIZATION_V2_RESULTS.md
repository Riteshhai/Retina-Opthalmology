# APTOS source optimization v2 â€” measured results

**VERIFIED COMPLETE:** the locked primary-seed schedule extension and saved-evidence audit. Overall scientific acceptance remains **IN PROGRESS**.

Execution source `a75229f92468a253e8f6c3e28d834571e25bd9e5` was committed before learning. [Fixed protocol](APTOS_OPTIMIZATION_V2_PROTOCOL.md). Aggregate SHA256: `66c49f12bf630493c9dc38180a73daeeae6a222523067f4ec262971b45d2310e`.

All three previous four-epoch runs selected their final epoch. That source-development finding motivated a new cosine trajectory of up to 15 epochs with patience four, restarting the same frozen-feature warm start. Only the original primary seed 2026 was extended for this CPU study. Budget and learning-rate trajectory both change; differences cannot be assigned solely to additional epochs. The previous three-seed results remain unchanged; no new seed variance is estimated.

Prior IDRiD outcome inspection is disclosed. No IDRiD images were used for fitting or scoring; earlier aggregate reports were hash-bound for preservation. The 351 reserved APTOS test images and competition test remain untouched. Production MobileNet/quality models and conformal sidecar remain unchanged.

This is 224px cropped-RGB parameter fine-tuning with frozen BatchNorm running statistics, horizontal flips and weighted cross-entropy. It differs from the paper's 384px procedure, and early stopping does not prove convergence.

## Locked optimization and grading

Completed **14** epochs / **1862** optimizer updates; selected epoch **10** by macro-F1 then lower unweighted NLL. Stop reason: `four_nonimproving_selection_epochs`. Selected at maximum epoch: `False`. Convergence remains unestablished.

All comparisons use the same reused 521 selection images. Temperature preserves the predicted grade; raw grading below avoids mixing different temperature fits.

| Classifier | Accuracy | Macro-F1 | QWK | Grade-3 recall | Grade-4 recall | Raw NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen-feature baseline | 78.69% | 0.60334 | 0.84400 | 23.08% | 55.00% | 0.68315 |
| Original four-epoch primary | 78.31% | 0.63349 | 0.86005 | 65.38% | 47.50% | 0.70350 |
| Extended primary | 80.81% | 0.65157 | 0.86726 | 26.92% | 67.50% | 0.82975 |

All five grade counts, precision/recall/F1, confusion matrices, referable AUROC/AUPRC, NLL/ECE/Brier and reliability bins remain in the aggregate. The source-grade supports are 262/51/142/26/40. No checkpoint is promoted to deployment.

| Epoch | Weighted training batch-loss mean | Selection F1 | Accuracy | Unweighted NLL | Backbone LR | Selected at that epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.02456 | 0.57965 | 74.86% | 0.68370 | 0.00010000 | True |
| 2 | 0.70641 | 0.53635 | 69.87% | 0.71851 | 0.00009892 | False |
| 3 | 0.54156 | 0.58365 | 75.82% | 0.71475 | 0.00009572 | True |
| 4 | 0.42859 | 0.62049 | 75.62% | 0.76951 | 0.00009055 | True |
| 5 | 0.30638 | 0.61453 | 81.00% | 0.72319 | 0.00008362 | False |
| 6 | 0.19619 | 0.61975 | 76.58% | 0.84874 | 0.00007525 | False |
| 7 | 0.15060 | 0.62420 | 79.08% | 0.68826 | 0.00006580 | True |
| 8 | 0.10267 | 0.62884 | 76.01% | 0.95266 | 0.00005567 | True |
| 9 | 0.06090 | 0.61504 | 79.27% | 0.86534 | 0.00004533 | False |
| 10 | 0.07299 | 0.65157 | 80.81% | 0.82975 | 0.00003520 | True |
| 11 | 0.05524 | 0.64747 | 81.57% | 0.80379 | 0.00002575 | False |
| 12 | 0.02712 | 0.63844 | 81.77% | 0.81930 | 0.00001738 | False |
| 13 | 0.02402 | 0.63587 | 81.57% | 0.81888 | 0.00001045 | False |
| 14 | 0.01531 | 0.62881 | 81.00% | 0.83439 | 0.00000528 | False |

The first epoch exactly reproduced all 521 saved logits from the previous primary run (maximum logit delta 0), before the cosine schedules diverged. Every epoch and unfavorable outcome is retained.

## Descriptive paired development changes

Fixed paired fingerprint-component percentile bootstrap: 2,000 resamples, seed 20261106, 95% intervals. Each resampled component retains every member and duplicate draw. No refitting, reselection or multiplicity correction occurs. These reused selection data and unidentified related views preclude fresh-test, patient-independent or clinical-superiority claims.

| Extended primary minus comparator | Accuracy delta [95% interval] | Macro-F1 delta [95% interval] |
| --- | --- | --- |
| Frozen-feature baseline | +0.02111 [-0.00769, +0.05019] | +0.04823 [-0.00692, +0.10555] |
| Four-epoch primary | +0.02495 [-0.00958, +0.05792] | +0.01808 [-0.04119, +0.07880] |

## Reused independent-role calibration diagnostics

All four descriptive paired intervals above include zero; these point differences
do not establish superiority on fresh data.

Temperature **2.94099575** was fitted on the original 262 temperature-role images. The disjoint original 254 conformal representatives fit score thresholds. Previously inspected calibration and selection data are reused, so these are empirical development diagnostics, not fresh calibration validation or patient/grade-conditional/external coverage guarantees.

All 14 fixed comparisons, both alpha levels and all five uniform seeds are retained, separately from classifier training seed 2026. Raw empty sets and common argmax fallback are distinct. All 521 observations and 507 fingerprint representatives remain in the aggregate.

| Alpha | Method | Uniform seed | Raw inclusion / size / empties | Fallback inclusion / size | Fallback grade inclusion 0 / 1 / 2 / 3 / 4 |
| ---: | --- | ---: | --- | --- | --- |
| 0.10 | lac | - | 89.25% / 1.296 / 0 | 89.25% / 1.296 | 96.95% / 84.31% / 90.14% / 46.15% / 70.00% |
| 0.10 | ordinal_greedy | - | 90.21% / 2.583 / 42 | 98.27% / 2.664 | 99.62% / 98.04% / 100.00% / 100.00% / 82.50% |
| 0.10 | ordinal_greedy_randomized | 2026 | 91.36% / 1.760 / 16 | 94.43% / 1.791 | 97.33% / 90.20% / 95.77% / 96.15% / 75.00% |
| 0.10 | ordinal_greedy_randomized | 2027 | 88.87% / 1.585 / 27 | 93.67% / 1.637 | 97.33% / 92.16% / 93.66% / 88.46% / 75.00% |
| 0.10 | ordinal_greedy_randomized | 2028 | 89.06% / 1.639 / 25 | 93.86% / 1.687 | 97.71% / 90.20% / 93.66% / 92.31% / 75.00% |
| 0.10 | ordinal_greedy_randomized | 2029 | 90.60% / 1.655 / 22 | 94.82% / 1.697 | 97.33% / 92.16% / 97.18% / 92.31% / 75.00% |
| 0.10 | ordinal_greedy_randomized | 2030 | 91.75% / 1.678 / 16 | 94.82% / 1.708 | 97.71% / 92.16% / 97.18% / 88.46% / 75.00% |
| 0.05 | lac | - | 94.05% / 1.591 / 0 | 94.05% / 1.591 | 97.71% / 90.20% / 96.48% / 65.38% / 85.00% |
| 0.05 | ordinal_greedy | - | 96.55% / 3.175 / 11 | 98.66% / 3.196 | 99.62% / 100.00% / 100.00% / 100.00% / 85.00% |
| 0.05 | ordinal_greedy_randomized | 2026 | 96.55% / 2.198 / 4 | 97.31% / 2.205 | 98.85% / 96.08% / 98.59% / 100.00% / 82.50% |
| 0.05 | ordinal_greedy_randomized | 2027 | 95.01% / 1.979 / 10 | 96.93% / 1.998 | 98.47% / 94.12% / 97.89% / 100.00% / 85.00% |
| 0.05 | ordinal_greedy_randomized | 2028 | 93.28% / 1.904 / 15 | 96.16% / 1.933 | 98.09% / 92.16% / 98.59% / 92.31% / 82.50% |
| 0.05 | ordinal_greedy_randomized | 2029 | 92.71% / 1.812 / 14 | 95.39% / 1.839 | 97.71% / 94.12% / 97.18% / 92.31% / 77.50% |
| 0.05 | ordinal_greedy_randomized | 2030 | 96.55% / 2.086 / 6 | 97.70% / 2.098 | 98.47% / 96.08% / 99.30% / 100.00% / 87.50% |

Higher marginal inclusion or wider ordinal intervals do not establish severe-grade coverage or useful triage. All denominators, set-size distributions and contiguity/empty counts remain inspectable.

At alpha .10, LAC includes only 46.15% of grade3 and 70.00% of grade4 images.
Deterministic ordinal raw inclusion 90.21%/size 2.583 becomes 98.27%/2.664 after
fallback; the enlargement is separate from the raw method. All five randomized
fallback replicates include only 75% of grade4 images despite mean marginal
inclusion 94.32%. The higher point macro-F1 does not resolve these severe-grade failures.

## Verification and preserved evidence

The independent saved-logit/math audit reconstructed all 14 epoch metrics and checkpoint rank, optimizer/LR schedule, order/flip RNG, selected/last parameter and BatchNorm states, temperature objective, four paired intervals, 14 conformal comparisons and 14,588 raw/fallback sets. No independent neural forward pass or refitting is claimed.

All 134 bound prior reports/figures, master copies, manifests and production/pilot artifacts matched. All 3,163 development image hashes and cache tensor bytes were checked. Independent pixel geometry was reconstructed on 10 train/selection plus 5 calibration images. Fingerprint components remain distinct from patient identities; source label-dependent quarantine and unknown related views constrain interpretation.

The full pre-execution suite passed 502 tests, zero failures/skips, 82.77s with genuine production-model/conformal integration. Ruff and Compose configuration passed. The subsequent array-ownership fix passed all 503 tests, zero failures/skips, 91.16s. Actual native JPEG single/batch/TTA/readiness/simulator and live-API Streamlit component checks passed, retaining all comparators, both alpha/population controls, all five uniform seeds and the fixed single primary/reuse/stopping captions. New container/browser proof is separate; current Docker engine availability is recorded in verification.json.

The execution logged a PyTorch warning for a read-only NumPy Boolean index view. After the complete measured audit, the runner was changed to own mask copies. An actual-data check confirmed numerically identical 262/254 indices and no warning; no retinal fitting/inference or scientific report was rerun. The measured execution source remains the pinned commit above, and verification records this later source change separately.

Runner wall time includes canonical input preparation and post-hoc diagnostics with concurrent read-only checks; it is not a controlled CPU speed benchmark. Derived figures were visually inspected. Private images, identifiers, manifests, logits, sets, caches and checkpoints remain local and ignored.

[Aggregate](../../reports/aptos_optimization_v2/evaluation.json), [independent audit](../../reports/aptos_optimization_v2/independent_audit.json), [verification](../../reports/aptos_optimization_v2/verification.json), [training/grading figure](../../reports/aptos_optimization_v2/training_and_grading.png), [uncertainty figure](../../reports/aptos_optimization_v2/conformal_diagnostics.png).

## Remaining requirements

Broader optimization replication, exact 384px paper reproduction, reliable quality supervision and useful validated triage remain open. Any fresh reserved-test evaluation needs its own locked protocol. Consumed IDRiD outcomes cannot tune final models/policies or become fresh confirmation. Formal manuscript drafting remains deferred.
