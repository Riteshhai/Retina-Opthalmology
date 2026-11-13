# Quality adaptation v1 — declared 5 November 2026

Status at declaration: no new real-image quality updates or outcomes observed.
This development study follows the weak natural-defect findings in the consumed
internal confirmation, repeated-seed and controlled-degradation reports. It is
an exploratory adaptation study, not independent confirmation, paper reproduction,
an external-quality evaluation or a deployment-selection procedure.

## Population and exclusions

Use only the frozen DeepDRiD official-training development manifest, SHA256
45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d.
Fit 848 images/212 patients/components; select on 168/42/42. Read the remaining
168/42 calibration metadata only for overlap guards; never open or predict its
pixels. All four known views stay together. No consumed 340-image confirmation,
IDRiD images/outcomes, APTOS images or reserved tests enter fitting or selection.
Official DeepDRiD source/terms remain as recorded in DATA_ACCESS_AND_TERMS.md.

All labels are complete, verified source annotations. Overall quality uses two
classes, **poor positive class 0**. Clarity and field-definition source codes
1,4,6,8,10 map to classes 0..4 (higher better). Artifact codes 0,1,4,6,8,10 map
to 0..5 (higher more artifact, 0 none). No derived disease/quality labels.

| Head | Fitting supports | Selection supports |
| --- | --- | --- |
| Overall quality | 438,410 | 98,70 |
| Clarity | 19,70,188,272,299 | 5,9,42,61,51 |
| Field definition | 3,33,130,576,106 | 1,6,21,116,24 |
| Artifact | 418,31,169,105,99,26 | 80,6,35,20,20,7 |

Retain every class and all denominators, especially the one-image field class.

## Six matched runs

Seeds 2026,2027,2028 each start both arms from that seed's historical selected
frozen-feature four-head MobileNet-v3-small state. Original report SHA256:
1ddb1d230a53dab0c75019e73654f2ab8d17313cc586a9c70efeff29fae4bb9e.
Warm checkpoint SHA256s, in seed order:

- 67368a38922423e38d541f96bfa7a49ce740d7b172eef7389073367cee6ffbc1
- 545c367d226951d63e9e0170407289ffa0474686f3355680a7c32510fd9aa9ca
- 09e19a910c0c6d9c80bb743aa47c2476d1b6091802421ea7210dd458ee052285

Arm `frozen` trains heads only; arm `adapted` trains heads plus every learnable
backbone parameter, including BatchNorm affine parameters. Both keep BatchNorm
running means/variances/counters fixed. All other choices are matched. Reproduce
each warm state's 168 selection probabilities first: maximum absolute delta
<=2e-5 for every head and all argmaxes identical, otherwise stop before learning.

Run exactly **15 epochs**, without early stopping, batch 16 (53 updates/epoch,
795/run,4,770 total). Every fitting image is used once/epoch, with independent
probability-.5 horizontal flip. Seed one explicit generator after cache/model
initialization; draw a permutation then all flips each epoch. The corresponding
arms must have identical order and flips. Canonical uint8 fundus crop/letterbox
224 caches apply geometry once; use established ImageNet normalization. CPU four
threads, channels-last, two decoding workers. No retinal timing extrapolation
from the preparation-only generated-tensor compute probe is a measured result.

AdamW: heads3e-4, adapted backbone1e-4,weight decay1e-4. CosineAnnealingLR T_max15,
eta_min1e-6, stepped after each epoch. Loss is the equal mean of four separately
class-weighted CE losses, with fitting-only weights N/(K*n_class). Record every
epoch's raw selection logits, metrics, order/flip hashes and learning rates.

Select among trained epochs1..15 using mean macro-F1 over all four heads, then
lower mean unweighted per-head NLL; keep the earliest exact tie. The warm epoch0
is a separate reference and cannot win a new trained checkpoint. The legacy
selector was overall-quality-only and legacy fitting used cached original+flip
features. Thus only adapted-minus-matched-frozen contrasts isolate adaptation;
warm-start differences combine additional changes and are descriptive.

Seed2026 is the fixed primary. Report all six runs, including adverse results.
Conditional means and sample SDs use all three seeds per arm; no winning-seed
deployment. Production checkpoints, conformal sidecar and triage policy stay fixed.

## Outcomes and uncertainty

Primary outcome: adapted-minus-frozen mean four-head macro-F1 for each seed.
Secondary: accuracy and macro-F1 for each head. Report all raw per-head NLL,
confusion matrices, precision, recall, F1/support, multiclass Brier, ECE with ten
equal-width confidence bins ([left,right), final right inclusive). Overall-quality
poor-class0 AUROC,AUPRC,sensitivity,specificity use the declared class direction.
Absent-denominator precision/recall/F1 is zero. No temperature/conformal fitting.

Reference majority point predictions choose the most frequent fitting class
(smallest class on ties) and report selection accuracy/F1/support only; do not
fabricate probabilities. Keep all legacy warm-start head outcomes for context.

For each seed, use common whole split_group_id bootstrap draws: 42 components,
2,000 replicates, numpy default_rng20261107, sample42 components with replacement,
retain every view, fixed full class universe, percentile2.5/97.5 intervals. Reuse
the same draw stream across all seeds. Retain all9 contrasts/seed (27 total),
without multiplicity adjustment. These descriptive intervals condition on reused
selection data and selected checkpoints; no fitting/reselection or population
independence guarantee. Do not claim a useful triage improvement from head metrics.

No new controlled-stress experiment is included in this study. Natural annotation
performance is its scope; deployment routing and independent quality validation
remain later work.

## Execution and audit

Run readiness without neural fitting/inference:
`python scripts/run_quality_adaptation.py`.
Commit tested source/config/protocol clean before
`python scripts/run_quality_adaptation.py --execute`.
`--execute --resume` accepts exact bindings only. Atomically commit one state per
epoch (model/optimizer/scheduler/RNG/history/all epoch logits/selected snapshot).
An interrupted epoch can repeat from its previous state; completed epochs cannot.
Seal final artifacts; never overwrite a sealed run or a differing measured report.

Bind source commit/file hashes, manifest/image/cache bytes, all historical warm
artifacts, both master copies, prior aggregate results and production artifacts.
Private paths, IDs, predictions and checkpoints stay ignored locally. Publish
aggregates only. Independently reconstruct all90 epoch metrics/selection ranks,
90 order/flip streams/schedules, all27 intervals, preserved hashes, parameter/BN
changes, canonical pixel hashes and sampled geometry. Verify native saved-results
API/dashboard and existing genuinely trained production inference, then run tests,
Ruff and available deployment checks. No container verification without a working
Docker engine. Overall QSART research acceptance remains IN PROGRESS.
