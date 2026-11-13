# Quality adaptation v1 — measured development results, 5 November 2026

Milestone **VERIFIED COMPLETE** for the declared six-run execution, independent
reconstruction and native integration. Overall QSART scientific acceptance
**IN PROGRESS**. This is reused source-development evidence, without a
production switch, fresh confirmation or validated triage claim.

## Frozen design and execution

Scientific source: `2d4243c977a0669776a5dc356d6d6e3166732a12` (clean before learning). All seeds
2026/2027/2028 completed both arms: 15 epochs and 795 optimizer updates each,
90 trained epochs and 4,770 updates in total. Seed 2026 remains the fixed primary.
Each pair shares the same historical warm state, fitting images, permutation,
flip stream, head optimizer and balanced four-head selector. Only the backbone
gradient/optimizer group differs; BatchNorm buffers remain fixed in both arms.
[Protocol](QUALITY_ADAPTATION_V1_PROTOCOL.md) and [aggregate](../../reports/quality_adaptation_v1/evaluation_v1_1.json).

Fit 848 images from 212 known patients/components; select on 168 images from 42 patients/components. Calibration
metadata guard grouping only; no calibration pixels, consumed confirmation,
IDRiD or APTOS images were opened for this study. Historical results and exact
master bytes stay unchanged. No temperature, conformal, policy or deployment fitting.

## Every seed and arm

| Seed | Arm | Selected epoch | Mean four-head F1 | Mean head NLL |
| --- | --- | ---: | ---: | ---: |
| 2026 | frozen | 5 | 0.51218942 | 0.97622343 |
| 2026 | adapted | 13 | 0.54652108 | 1.02045741 |
| 2027 | frozen | 6 | 0.50938879 | 0.94485393 |
| 2027 | adapted | 8 | 0.53661531 | 0.99319272 |
| 2028 | frozen | 10 | 0.50524589 | 0.94539796 |
| 2028 | adapted | 8 | 0.54023040 | 0.97837088 |

All trained epochs 1..15 were eligible; warm epoch 0 is a reference only.
Legacy warm comparisons also change the selector and fitting procedure and
cannot isolate adaptation. The matched frozen arm provides that comparison.

| Arm | Conditional mean four-head F1 | Sample SD across three seeds |
| --- | ---: | ---: |
| frozen | 0.50894137 | 0.00349332 |
| adapted | 0.54112226 | 0.00501275 |

These SDs condition on one split and the historical selected warm states;
they do not estimate patient-population or full-pipeline uncertainty.

| Seed | Adapted minus frozen mean F1 | Descriptive 95% interval |
| --- | ---: | --- |
| 2026 | +0.03433166 | [-0.01349020, +0.06511460] |
| 2027 | +0.02722652 | [-0.02551619, +0.06675067] |
| 2028 | +0.03498451 | [-0.01528876, +0.07434836] |

3/3 primary-outcome intervals include zero. All 27 declared primary/secondary
intervals remain in the aggregate. Each uses 2,000 common whole-component draws
(42 components, bootstrap seed 20261107), fixed selected checkpoints and full class sets.
No refitting/reselection or multiplicity correction; these are descriptive
selection-reuse intervals, not confirmatory superiority tests.

## Primary seed: all four heads and references

| Head | Warm F1 | Matched frozen F1 | Adapted F1 | Fitting-majority F1 | Frozen accuracy | Adapted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall quality | 0.75854063 | 0.72478632 | 0.77222777 | 0.36842105 | 0.72619048 | 0.77380952 |
| Clarity | 0.45767151 | 0.54032995 | 0.52877193 | 0.09315068 | 0.51190476 | 0.55952381 |
| Field definition | 0.42714048 | 0.38852018 | 0.49275666 | 0.16338028 | 0.61309524 | 0.74404762 |
| Artifact | 0.25120425 | 0.39512122 | 0.39232795 | 0.10752688 | 0.52380952 | 0.55357143 |

| Head | Frozen NLL | Adapted NLL | Primary F1 difference [95% interval] |
| --- | ---: | ---: | --- |
| Overall quality | 0.54041670 | 0.63230745 | +0.04744145 [+0.00354646, +0.09594398] |
| Clarity | 1.12458903 | 1.16960528 | -0.01155802 [-0.16630372, +0.08132497] |
| Field definition | 0.94970557 | 0.99388108 | +0.10423648 [+0.02920401, +0.16698554] |
| Artifact | 1.29018241 | 1.28603582 | -0.00279326 [-0.08984864, +0.07647203] |

Primary clarity and artifact macro-F1 decrease despite a higher mean four-head
score. Mean raw NLL worsens in all three pairs. Mean ECE increases in every
head: overall quality .03969 to .11181, clarity .09625 to .19145, field definition
.05884 to .12164 and artifact .10227 to .16871. Poor-quality AUROC falls in all
three pairs; primary .82201 to .80714. A selected F1 increase does not establish
better probability calibration or downstream triage performance.

Overall-quality poor-positive class is 0 throughout. All AUROC/AUPRC,
poor sensitivity/specificity, ten-bin ECE, Brier and confusion matrices
are retained for every run. Majority references are fitting-only point
predictors and have no fabricated probability/NLL.

## Sparse classes and remaining failures

| Head | Fitting supports | Selection supports |
| --- | --- | --- |
| Overall quality | 438, 410 | 98, 70 |
| Clarity | 19, 70, 188, 272, 299 | 5, 9, 42, 61, 51 |
| Field definition | 3, 33, 130, 576, 106 | 1, 6, 21, 116, 24 |
| Artifact | 418, 31, 169, 105, 99, 26 | 80, 6, 35, 20, 20, 7 |

Field-definition class 0 has only three fitting images and one selection image.
Its measured recall for each run is retained below; one outcome cannot
support a reliable rare-class generalization claim.

| Seed | Arm | Field class 0 recall (selection n=1) | Artifact class 1 recall (n=6) |
| --- | --- | ---: | ---: |
| 2026 | frozen | 0.00000000 | 0.16666667 |
| 2026 | adapted | 0.00000000 | 0.00000000 |
| 2027 | frozen | 0.00000000 | 0.00000000 |
| 2027 | adapted | 0.00000000 | 0.00000000 |
| 2028 | frozen | 0.00000000 | 0.00000000 |
| 2028 | adapted | 0.00000000 | 0.00000000 |

No new controlled-degradation experiment was included. Earlier compression
response failures are not resolved by these natural-annotation metrics.
Independent quality supervision/validation and a locked subsequent stress
comparison remain needed before considering deployment.

## Independent reconstruction and software evidence

- All 1,016 source SHA256s/cache bytes, 26 independent geometry samples and 171 prior artifacts match.
- Reconstructed all 90 trained and 6 warm epoch metric sets, selected ranks,
  order/flip streams and schedules; verified optimizer/RNG snapshot consistency.
  Selected weights reproduce their 168 exported selection logits for all six runs.
- Independently vectorized component bootstrap reproduces all 27 intervals;
  per-arm means/SDs, majority references and original warm probabilities match.
- All 521 local tests passed 128.16s before real learning, zero failures/skips,
  with genuine production-model/conformal integration. All 18 targeted tests
  passed 47.32s. The scientific source was fixed throughout all six runs.
- Afterwards, separated the training-seed label from bootstrap seed 20261107
  in paired-row metadata. The original report is preserved; version 1.1 and
  [metadata_correction.json](../../reports/quality_adaptation_v1/metadata_correction.json) bind both hashes. No models, epoch choices, metrics
  or interval values were recomputed. All 522 post-correction tests passed
  in 110.17s with zero failures/skips; the independent audit verifies the exact field-only change.
- Ruff, Compose configuration and whitespace checks pass. Actual native JPEG
  single/batch/TTA/readiness/simulator/overload checks pass; the saved-results
  explorer exposes 17 records and every seed/head with all supports/recalls.
- Both derived figures were visually inspected. Docker Linux engine remains
  unavailable; no new browser-rendering or container-execution claim.

Scientific aggregate SHA256: `b21f0305d3d2c19392d91434b7b4dedaed65c9329029b84939143ca8497f2fd8`.
Source/config/protocol, measured evidence and publication provenance remain
separate. Private identifiers/logits/checkpoints stay local and ignored.
Formal manuscript drafting stays deferred.

Derived figures: [training and head metrics](../../reports/quality_adaptation_v1/training_and_head_metrics.png), [class recalls and paired changes](../../reports/quality_adaptation_v1/class_recall_and_paired_changes.png).
