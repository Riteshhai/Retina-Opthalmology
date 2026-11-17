# Locked IDRiD external-domain evaluation v1 — results

**VERIFIED COMPLETE:** the first locked four-model external-domain evaluation and
independent saved-logit/mathematical reconstruction. Overall scientific acceptance
remains **IN PROGRESS**. Execution source `e447a21b55b82ca47d1e3b91caf76bca61f23699` and the
[protocol](IDRID_EXTERNAL_V1_PROTOCOL.md) were committed before the first external
forward pass. Aggregate SHA256: `812c82d7771752fa00d076e067e6d1692da4a414c0b1207c8e6df2f21ec35282`.

IDRiD model outcomes are now **CONSUMED**. Later reuse cannot be described as fresh
confirmation or silently used for final model/policy tuning. Official/internal
APTOS reserved-test inference remains untouched. Production models remain unchanged.

## Locked design and limitations

Four previously selected EfficientNet-B0 states were evaluated: frozen APTOS head
baseline and all three fine-tuning seeds 2026/2027/2028. Primary seed 2026 and main 505
fingerprint representatives were fixed beforehand. Full 516 and filtered 513 views
remain mandatory sensitivity populations. No classifier, source temperature or
conformal threshold was fitted or selected on IDRiD.

Every external JPEG was checked against its frozen SHA before canonical 224px caching.
Deterministic eval-mode CPU inference used four threads, batch 16 and channels-last.
All four ten-image development preflights passed; checkpoint/parameter/buffer
states remained unchanged. These are short source-trained models, not exact
384px base-paper optimization or patient-level/clinical validation.

The official-source confirmation and original audit/review remain preserved.
Three unresolved APTOS perceptual candidates remain in full 516 and are conservatively
flagged in the other views. Four source-grade-conflicting components remain in
all views. Representatives are not patients, and unknown related views, labels,
case mix and other dataset differences prevent a causal attribution of failures.
The 505-image population differs from the paper's reported 455 cohort.

## Classification against supplied source labels

Temperature preserves the predicted grade; it changes probability diagnostics.
All populations are shown, without picking the most favorable one.

| Model | Population | Images | Accuracy | Macro-F1 | QWK | Grade-3 recall | Grade-4 recall | Raw NLL -> source-T NLL | ECE10 after source T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Frozen-feature baseline | Full source | 516 | 43.41% | 0.36355 | 0.65095 | 5.38% | 62.90% | 2.05525 -> 1.44499 | 0.18461 |
| Frozen-feature baseline | Cross-source filtered | 513 | 43.47% | 0.36441 | 0.64909 | 5.43% | 62.90% | 2.05516 -> 1.44419 | 0.18457 |
| Frozen-feature baseline | Filtered representatives | 505 | 43.37% | 0.36195 | 0.65159 | 5.49% | 63.33% | 2.04994 -> 1.44200 | 0.18489 |
| Fine-tune 2026 (primary) | Full source | 516 | 43.22% | 0.41555 | 0.65319 | 43.01% | 43.55% | 1.81671 -> 1.38391 | 0.17710 |
| Fine-tune 2026 (primary) | Cross-source filtered | 513 | 43.27% | 0.41652 | 0.65116 | 43.48% | 43.55% | 1.82001 -> 1.38499 | 0.17734 |
| Fine-tune 2026 (primary) | Filtered representatives | 505 | 42.97% | 0.41359 | 0.65164 | 43.96% | 43.33% | 1.81767 -> 1.38231 | 0.18209 |
| Fine-tune 2027 | Full source | 516 | 44.38% | 0.40747 | 0.67180 | 25.81% | 62.90% | 1.76904 -> 1.35456 | 0.15474 |
| Fine-tune 2027 | Cross-source filtered | 513 | 44.44% | 0.40859 | 0.67016 | 26.09% | 62.90% | 1.76725 -> 1.35286 | 0.15463 |
| Fine-tune 2027 | Filtered representatives | 505 | 44.36% | 0.40590 | 0.67140 | 26.37% | 61.67% | 1.75829 -> 1.34729 | 0.15619 |
| Fine-tune 2028 | Full source | 516 | 37.02% | 0.35838 | 0.64442 | 19.35% | 66.13% | 2.22264 -> 1.64286 | 0.28012 |
| Fine-tune 2028 | Cross-source filtered | 513 | 37.04% | 0.35883 | 0.64238 | 19.57% | 66.13% | 2.22842 -> 1.64556 | 0.28089 |
| Fine-tune 2028 | Filtered representatives | 505 | 37.03% | 0.35788 | 0.64388 | 19.78% | 65.00% | 2.23095 -> 1.64652 | 0.28174 |

Primary 2026/main 505 accuracy is **42.97%**, macro-F1 **0.41359**,
and grade-3/4 recalls **43.96%/43.33%**.
The three fine-tuning macro-F1 values average 0.39246 with SAMPLE SD
0.03019, conditional on their common selected warm start.
The corresponding APTOS primary selection F1 was .63349; these are different
cohorts, not a paired source-to-target treatment comparison or a causal estimate
of domain shift. Poor external performance remains visible.

All five grade supports, precision/recall/F1, confusion matrices, referable
AUROC/AUPRC, Brier and reliability bins remain in the aggregate. The main 505
supplied-grade counts are 167/23/164/91/60. No favorable classifier seed is promoted.

## Descriptive paired baseline changes

Fixed paired fingerprint-component bootstrap: 2,000 resamples, seed 20261105,
95% percentile intervals, no refitting or multiplicity adjustment. Components are
not patients, and annotation uncertainty is separate from these conditional
intervals. These estimates do not establish clinical superiority.

| Fine-tuning model | Population | Accuracy delta [lower,upper] | Macro-F1 delta [lower,upper] |
| --- | --- | --- | --- |
| Fine-tune 2026 (primary) | Full source | -0.00194 [-0.04458,+0.04086] | +0.05200 [+0.00922,+0.09196] |
| Fine-tune 2026 (primary) | Cross-source filtered | -0.00195 [-0.04669,+0.04256] | +0.05210 [+0.00910,+0.09449] |
| Fine-tune 2026 (primary) | Filtered representatives | -0.00396 [-0.04950,+0.03960] | +0.05164 [+0.00889,+0.09468] |
| Fine-tune 2027 | Full source | +0.00969 [-0.02713,+0.04699] | +0.04392 [+0.00576,+0.08410] |
| Fine-tune 2027 | Cross-source filtered | +0.00975 [-0.02745,+0.04845] | +0.04417 [+0.00502,+0.08317] |
| Fine-tune 2027 | Filtered representatives | +0.00990 [-0.02772,+0.04757] | +0.04395 [+0.00531,+0.08166] |
| Fine-tune 2028 | Full source | -0.06395 [-0.10291,-0.02519] | -0.00517 [-0.04121,+0.03183] |
| Fine-tune 2028 | Cross-source filtered | -0.06433 [-0.10506,-0.02519] | -0.00558 [-0.04290,+0.03168] |
| Fine-tune 2028 | Filtered representatives | -0.06337 [-0.10495,-0.02376] | -0.00407 [-0.04244,+0.03414] |

The primary macro-F1 difference from the frozen baseline is positive under this
fixed-component descriptive procedure, while its accuracy interval spans zero.
Severe-grade tradeoffs remain: primary grade-3 recall exceeds the baseline, but
grade-4 recall is lower. No deployment or clinical benefit follows from that result.

## Frozen conformal transfer — every primary-model row

All 56 model/method/alpha/uniform comparisons are retained in the full report and
explorer. The table below is the PREDECLARED primary 2026/main 505 subset. Every
artifact and temperature value is reused unchanged from APTOS; no target recalibration
occurs. External uniforms use SeedSequence child 2, independent of source child 0
calibration/child 1 selection, shared across classifiers/alpha. Uniform seeds are
not additional classifier-training runs.

| Alpha | Method | Uniform seed | Raw inclusion / mean size / empties | Fallback inclusion / mean size | Fallback grade0 / grade1 / grade2 / grade3 / grade4 inclusion |
| ---: | --- | --- | --- | --- | --- |
| 0.10 | lac | none | 70.69% / 1.931 / 0 | 70.69% / 1.931 | 64.07% / 82.61% / 76.83% / 67.03% / 73.33% |
| 0.10 | ordinal_greedy | none | 96.04% / 3.461 / 0 | 96.04% / 3.461 | 91.62% / 100.00% / 98.78% / 100.00% / 93.33% |
| 0.10 | ordinal_greedy_randomized | 2026 | 85.54% / 2.685 / 1 | 85.74% / 2.687 | 81.44% / 86.96% / 86.59% / 95.60% / 80.00% |
| 0.10 | ordinal_greedy_randomized | 2027 | 80.40% / 2.442 / 5 | 81.39% / 2.451 | 74.85% / 86.96% / 82.93% / 95.60% / 71.67% |
| 0.10 | ordinal_greedy_randomized | 2028 | 82.77% / 2.471 / 3 | 83.17% / 2.477 | 79.64% / 86.96% / 82.32% / 94.51% / 76.67% |
| 0.10 | ordinal_greedy_randomized | 2029 | 82.97% / 2.487 / 3 | 83.56% / 2.493 | 78.44% / 91.30% / 84.76% / 95.60% / 73.33% |
| 0.10 | ordinal_greedy_randomized | 2030 | 84.75% / 2.610 / 2 | 85.15% / 2.614 | 80.24% / 82.61% / 87.20% / 94.51% / 80.00% |
| 0.05 | lac | none | 80.99% / 2.499 / 0 | 80.99% / 2.499 | 76.05% / 86.96% / 85.98% / 74.73% / 88.33% |
| 0.05 | ordinal_greedy | none | 96.44% / 3.766 / 0 | 96.44% / 3.766 | 91.62% / 100.00% / 100.00% / 100.00% / 93.33% |
| 0.05 | ordinal_greedy_randomized | 2026 | 93.66% / 3.172 / 0 | 93.66% / 3.172 | 90.42% / 95.65% / 95.73% / 98.90% / 88.33% |
| 0.05 | ordinal_greedy_randomized | 2027 | 88.91% / 2.873 / 3 | 89.50% / 2.879 | 84.43% / 91.30% / 91.46% / 98.90% / 83.33% |
| 0.05 | ordinal_greedy_randomized | 2028 | 92.67% / 2.994 / 1 | 92.87% / 2.996 | 89.22% / 95.65% / 93.90% / 97.80% / 91.67% |
| 0.05 | ordinal_greedy_randomized | 2029 | 91.29% / 2.915 / 0 | 91.29% / 2.915 | 87.43% / 91.30% / 92.68% / 97.80% / 88.33% |
| 0.05 | ordinal_greedy_randomized | 2030 | 90.10% / 2.921 / 1 | 90.30% / 2.923 | 86.23% / 95.65% / 91.46% / 98.90% / 83.33% |

At alpha .10, primary LAC inclusion is only 70.69% with mean size 1.931.
Deterministic ordinal inclusion is 96.04% with size 3.461: higher inclusion comes
with broader sets, not proof of a useful triage policy. Across all five randomized
ordinal draws, raw mean inclusion is 83.29%/size 2.539; common fallback gives
83.80%/2.545. Neither attains 90% aggregate inclusion in this view.
At alpha .05, deterministic marginal inclusion exceeds 95%, yet grade0 and grade4
inclusion remain 91.62% and 93.33%. No conditional/patient coverage guarantee or
universal ordinal-method advantage is established. Raw empties, size histograms,
contiguity and all three populations remain in the full report.

## Source-annotation sensitivity

For each original fingerprint component let A contain ALL its supplied DR grades.
For a prediction set S, report lower event A subset S and upper event A intersects S;
a singleton S gives the corresponding accuracy envelope. This assumes some label
in A is appropriate for that observation; A need not contain the clinical truth,
and perceptual components may contain distinct views. These are hypothetical
annotation-sensitivity envelopes, not confidence intervals or corrected grades.

| Population | Primary as-provided accuracy | Conditional annotation lower / upper | Ambiguous observations |
| --- | ---: | --- | ---: |
| Full source | 43.22% | 42.44% / 43.80% | 9 |
| Cross-source filtered | 43.27% | 42.50% / 43.86% | 9 |
| Filtered representatives | 42.97% | 42.77% / 43.37% | 4 |

Every raw/fallback set also retains this envelope. Unambiguous-component grading
is separately labeled as a label-dependent diagnostic stratum; it is never
substituted for the fixed main population. Original source grades remain unchanged.

## Verification and preserved artifacts

Independent reconstruction verified every saved raw/calibrated grading view,
all 56 comparisons/57,792 raw-fallback individual sets, 360 annotation envelopes,
18 paired intervals, seed summaries and fixed primary/consumed flags. All 516
external hashes and cache tensor bytes were checked; 25 deterministic label-
stratified images were independently recropped/resized. State hashes matched all
four original checkpoints. This is saved-logit mathematical reconstruction,
without a second neural forward pass or any independent refitting.

All 31 previous master, production and APTOS/IDRiD audit/review artifacts matched.
Images, per-image logits, uniforms, sets and caches stay private/ignored. The exact
report, sources, checkpoints, transforms and software versions are hash-bound.
The full pre-execution suite passed 491 tests, zero failures/skips, 84.64s with
actual production-model/conformal integration. Ruff and Compose configuration
passed. The initial 490-test run passed before exact export/replay regression was added.

Actual native HTTP and live-API Streamlit AppTest passed: 15 records, all 4 models,
all 3 population controls, alpha and all 5 uniform draws, consumed-status/primary
captions and unchanged production card/sidecar. Actual JPEG single/batch/TTA,
readiness and simulator checks passed after restart. New container and browser-
rendering proof remain separate; the current Docker engine is unavailable.
Figures were generated from the measured aggregate and visually inspected.

[Full aggregate](../../reports/idrid_external_v1/evaluation.json),
[independent audit](../../reports/idrid_external_v1/independent_audit.json),
[verification](../../reports/idrid_external_v1/verification.json),
[grading figure](../../reports/idrid_external_v1/external_grading.png),
[uncertainty figure](../../reports/idrid_external_v1/external_conformal.png).

## Remaining requirements

IDRiD outcomes are now consumed. Future development may use source development
partitions, but not IDRiD to tune final models, calibration or policy thresholds.
Any later external reuse must disclose prior outcome inspection. Stronger source
optimization, reliable quality supervision, broader independent validation and
useful validated triage remain open. This experiment does not switch deployment,
correct clinical labels or satisfy exact-paper/clinical acceptance. Formal
manuscript drafting remains deferred.
