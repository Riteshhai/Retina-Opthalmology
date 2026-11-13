# APTOS source optimization v2 — declared 4 November 2026

This is a source-development extension of the completed four-epoch pilot. All
three previous runs selected their final epoch, so that source evidence supports
testing a longer schedule. Previously consumed IDRiD outcomes have been inspected
and are disclosed, but they do not select this configuration. Do not infer on or
tune to IDRiD. Preserve the reserved 351-image APTOS test and competition test.

## Fixed design

Extend only the original predeclared PRIMARY seed 2026 for this bounded CPU study.
The earlier three seeds remain unchanged and fully reported in their own study.
This extension does not estimate new across-seed variance or represent the entire
optimization pipeline's uncertainty. A favorable old or new seed is never substituted.

Start again from the SAME frozen-feature EfficientNet-B0 warm start, SHA256
`52574e61db4b68308348aab3794bcc64c55f6d47b0d02932955286c35d190407`.
Do not resume the four-epoch cosine trajectory with a changed scheduler.
The new trajectory has maximum 15 epochs, cosine T_max 15, eta_min .000001,
patience four, and selection among trained epochs by macro-F1 then lower
unweighted NLL. Epoch zero remains a separate frozen-feature comparator.

All other pilot settings are fixed: cropped RGB 224px, ImageNet normalization,
batch 16, AdamW weight decay .0001, initial backbone/head rates .0001/.0003,
training-only Bernoulli(.5) horizontal flips and weighted CE N/(5*n_class).
All learnable parameters adapt; BatchNorm running buffers remain fixed. Four
CPU threads, channels-last memory and two canonical-cache workers remain fixed.
This is not the paper's 384px/no-crop/expanded-augmentation procedure or proof of
convergence. Early stopping describes a selection rule, not mathematical convergence.

## Frozen inputs and exclusion rules

Reuse the exact original 2,115 fitting / 521 selection / 527 calibration images,
fingerprint components and quarantine, mapping identities to QSART_APTOS_ROOT.
The development SHA is
`681799c7ee5f04e988dc607f6cf296fd882c5f55f2603dcd0a2ccd8148951fab`.
The prior pilot report SHA is
`8c6f50e93db412fc845600a4e368094c46fce041109806e37819dcd6bb70c5f4`.
Verify each development image against its original SHA before any cache reuse.
Only train/validation pixels enter the fitting cache; calibration is separate.
No test, confirmation, external or IDRiD record is accepted by the training path.
Fingerprint components are not patient identities; no quality labels are invented.

## Fixed comparisons and post-hoc diagnostics

Retain every epoch's metrics/logits, order/flip counts, optimizer steps, learning
rates, selected and last weights, BN/parameter hashes and RNG/scheduler state.
Evaluate the selected model on all 521 reused selection images, reporting all
five grade supports/recalls, confusion matrices, accuracy, macro-F1, QWK,
NLL/ECE/Brier and referable AUROC/AUPRC. Report the stop reason and whether the
selected epoch reaches the declared cap; never assert convergence from these alone.

Pair the new selected predictions with BOTH the frozen-feature baseline and the
original primary seed's four-epoch model. For accuracy/macro-F1, use fixed paired
fingerprint-component percentile bootstrap, 2,000 draws, seed 20261106, 95%
intervals, no retraining/reselection/multiplicity adjustment. These are descriptive
development intervals after checkpoint and study-design reuse, not fresh-test
superiority. Preserve adverse severe-grade, accuracy and calibration changes.

After training, fit temperature on the same independent-role 262 images; score
thresholds use the same disjoint 254 conformal representatives. This reuses
previously inspected calibration data and therefore gives empirical development
diagnostics, not fresh calibration validation. Retain LAC, deterministic ordinal
and ALL five randomized ordinal seeds 2026–2030 at alpha .10/.05 (14 comparisons).
Use the existing separate calibration/selection uniform streams and common draws
across alpha. Report raw empties and common argmax fallback separately on all 521
images and 507 representatives, with all grade denominators and set-size details.

## Execution, audit and preservation

Commit tested code/config/protocol before actual learning. Bind exact source,
original/runtime manifests, source bytes, warm start, previous study and every
preserved public report/figure and selected production/pilot checkpoint. Default
runner mode performs readiness only. Exact-provenance resume restores completed
epoch/optimizer/scheduler/RNG state; changed configuration or namespace is rejected.
Use a distinct `aptos-optimization-v2` model-version namespace. Never overwrite an
old report, checkpoint, manifest or specification. Do not modify execution sources
while the run is active.

Verify all 521 warm-start predictions against the old baseline before learning.
Independently reconstruct epoch selection, raw/calibrated grading, temperature
objective, paired intervals, all 14 comparisons/14,588 raw-fallback sets, source
and cache bindings and parameter/BN changes. Keep images, identifiers, per-image
scores/sets and weights private. Publish aggregate-only reports and derived figures.
Verify API/dashboard saved results and existing production readiness; deployment
models remain unchanged. New Docker proof is separate from native execution.

Overall project and exact-paper/clinical acceptance remain IN PROGRESS. Formal
manuscript drafting stays deferred. Further development must not tune to consumed
IDRiD outcomes; fresh testing remains reserved until a separately locked protocol.
