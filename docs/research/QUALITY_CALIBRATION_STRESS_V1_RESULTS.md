# Quality calibration and stress v1 — measured results, 5 November 2026

Milestone **VERIFIED COMPLETE** for the declared execution, independent audit
and native integration. Overall QSART research acceptance **IN PROGRESS**.
No classifier, production calibration, conformal artifact or policy was replaced.

Scientific execution source: `42ae8621649b11d666f7b5e3e8516b0ac3a5a812` (committed clean before inference/fitting).
[Protocol](QUALITY_CALIBRATION_STRESS_V1_PROTOCOL.md), [aggregate](../../reports/quality_calibration_stress_v1/evaluation.json),
[independent audit](../../reports/quality_calibration_stress_v1/independent_audit.json) and
[verification](../../reports/quality_calibration_stress_v1/verification.json).

## Roles and frozen candidates

All six selected quality-adaptation checkpoints were fixed in advance: training
seeds 2026/2027/2028 with frozen/adapted arms. Primary remains seed 2026 adapted.
Fit 24 head temperatures on 168 clean calibration images/42 patients; evaluate
clean selection on a disjoint 168/42 role. The calibration role was excluded from
these models fitting/selection but used historically with older models. Selection
repeatedly chose/evaluated checkpoints. These are development-reused diagnostics,
not fresh confirmation or external/clinical validation.

Only 336 source images were decoded. No fitting pixels, consumed confirmation,
IDRiD or APTOS images entered. Single-pass seven-condition selection evaluation
plus clean calibration gives 8,064 model-image observations; no classifier updates.
Private raw logits, pixel caches and temperature sidecars remain local/ignored.

## All scalar fits and clean selection outcomes

| Model | Overall-quality T | Clarity T | Field T | Artifact T | Mean NLL raw | Mean NLL scaled | Paired change [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quality_seed2026_frozen | 1.141197 | 0.844740 | 1.539534 | 1.007648 | 0.97622343 | 0.98753220 | +0.01130877 [-0.00221220, +0.02430387] |
| quality_seed2026_adapted | 2.444631 | 1.796584 | 2.942419 | 1.778477 | 1.02045741 | 0.87411869 | -0.14633872 [-0.21830587, -0.07930656] |
| quality_seed2027_frozen | 1.317656 | 0.999174 | 1.657212 | 1.081683 | 0.94485393 | 0.94676748 | +0.00191355 [-0.01732876, +0.01852673] |
| quality_seed2027_adapted | 2.299267 | 1.764955 | 2.748969 | 1.855402 | 0.99319272 | 0.87189609 | -0.12129664 [-0.18814126, -0.05996767] |
| quality_seed2028_frozen | 1.189830 | 0.983901 | 1.630383 | 1.049430 | 0.94539796 | 0.95034708 | +0.00494912 [-0.01341973, +0.02070306] |
| quality_seed2028_adapted | 2.105397 | 1.688819 | 2.476349 | 1.682018 | 0.97837088 | 0.87377678 | -0.10459410 [-0.16312263, -0.05029305] |

Primary mean four-head selection NLL falls 1.02045741 to .87411869;
change -.14633872, descriptive 95% interval[-.21830587,-.07930656]. Each adapted
model has lower point NLL; each frozen model has higher point NLL. All six fits
are retained rather than choosing raw/scaled probabilities from selection results.

| Arm | Mean NLL raw ± sample SD | Mean NLL scaled ± sample SD |
| --- | --- | --- |
| frozen | 0.95549177 ± 0.01795620 | 0.96154892 ± 0.02257325 |
| adapted | 0.99734034 ± 0.02134762 | 0.87326385 ± 0.00119679 |

These conditional SDs use three historical initializations/checkpoints on one
split. Temperature-fit metrics are in-sample, clearly separate in the aggregate/UI.

## Primary head diagnostics and adverse results

| Head | Raw NLL | Scaled NLL | Raw ECE | Scaled ECE | Raw Brier | Scaled Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall quality | 0.63230745 | 0.55139691 | 0.13266152 | 0.08222392 | 0.38300585 | 0.36942481 |
| Clarity | 1.16960528 | 1.01258640 | 0.20639893 | 0.08311642 | 0.63826759 | 0.58170722 |
| Field definition | 0.99388108 | 0.84827444 | 0.12950363 | 0.17579096 | 0.43207582 | 0.45713615 |
| Artifact | 1.28603582 | 1.08421701 | 0.19020135 | 0.08675052 | 0.60102781 | 0.55371494 |

Primary field-definition ECE worsens .12950 to .17579 and Brier .43208 to
.45714 despite lower NLL. Temperature scaling is optimized for calibration-role
NLL and does not ensure every score or selection diagnostic improves.

Every clean confusion matrix, accuracy, macro-F1 and rare-class recall stays
unchanged. Primary overall-quality accuracy remains .77380952/F1 .77222777,
and poor-positive class0 AUROC remains .80714286. Temperature scaling does not
repair discrimination: field class0 remains 0/1 in every model and artifact
class1 remains 0/6 in every adapted model. All denominators remain reported.

## Stress responses: all conditions for the fixed primary model

These altered images have **no new quality/defect ground truth**. Their outputs
are score responses, never corrupted quality accuracy/NLL/ECE. Expected-class
summaries use normalized ordinal codes, not physical/clinical severity units.
Perturbations occur after one 224px canonical crop and do not emulate all
acquisition defects. JPEG is bound to the recorded Pillow/libjpeg environment.

| Condition | Raw poor-score change | Scaled poor-score change | Scaled descriptive 95% interval |
| --- | ---: | ---: | --- |
| clean | +0.00000000 | +0.00000000 | [+0.00000000, +0.00000000] |
| blur_1 | +0.03885024 | +0.03299745 | [+0.01082997, +0.05522950] |
| blur_2 | +0.12151375 | +0.08822287 | [+0.04974196, +0.12779639] |
| exposure_0_5 | +0.23175253 | +0.16380667 | [+0.13729126, +0.19027525] |
| exposure_1_5 | -0.02246590 | -0.01155788 | [-0.03853100, +0.01755112] |
| jpeg_50 | +0.03824332 | +0.03070015 | [+0.00413430, +0.05660125] |
| jpeg_20 | +0.07134633 | +0.04371024 | [+0.00771073, +0.07743241] |

The predeclared blur2/JPEG20 scaled primary changes are +.08822287 and
+.04371024. These are probability shifts without newly labeled detection truth.
Larger shifts do not establish better recapture, diagnosis or clinical outcomes.

The primary scaled adapted-minus-frozen response contrast is -.12577232 under
blur2 (interval[-.15538536,-.09755179]): adaptation has a smaller poor-score
response there. JPEG20 contrast +.01490227 has interval[-.00818655,.03540913],
including zero. Raw contrasts and all other heads/conditions/seeds remain in
the aggregate. Calibrated contrasts also reflect separately fitted temperatures.

## Uncertainty and independent reconstruction

All 486 declared intervals and 48 clean zero controls are retained. Each uses
2,000 common whole-component draws over 42 selection groups, bootstrap seed 20261108,
with every view and fixed model/temperature. There is no refitting/reselection
or multiplicity correction. These are descriptive conditional development
intervals, not confirmatory superiority or coverage guarantees.

- Independent reconstruction passed all 96 clean head metric views and 534 interval/control records.
- All 24 temperature objectives were compared with an independent bounded optimizer; maximum NLL gap 3.99671384876e-09. Scientific temperatures were not replaced.
- All 8,064 neural outputs replay bit-for-bit; all model parameters/buffers remain fixed.
- All 336 source hashes/eight pixel caches, 26 independent geometry images/104 transform cells and 234 historical/master/production artifacts match.
- All 536 local tests passed in 100.90s, zero failures/skips, with genuine production/conformal integration; 14 targeted tests passed in 14.06s. The scientific/API/UI source stays fixed.
- Ruff, Compose configuration and whitespace checks pass. Real native JPEG/batch/TTA/readiness/simulator/overload checks pass after restart.
- Actual HTTP/AppTest verifies 18 records, 48 clean model/head/role and 48 stress model/state/response views, including fixed-primary and absent-corrupted-truth captions.
- Both derived figures were visually inspected. Docker Linux engine remains unavailable; no new browser/container proof.

Scientific aggregate SHA256: `906322ccd6a28a5a330a935573af6ac775de5e191eafa474ebd28f96c04b6bc3`.
No production replacement follows. Next work is independent quality-data access
and label validation before broader deployment or utility proposals. Formal
manuscript drafting stays deferred.

Figures: [clean calibration](../../reports/quality_calibration_stress_v1/calibration_diagnostics.png),
[primary score responses](../../reports/quality_calibration_stress_v1/stress_responses.png).
