# 384-pixel APTOS development results

The fixed ImageNet-initialized EfficientNet-B0 run completed 13 epochs and 3,445
optimizer updates on 9 November 2026. The accuracy rule selected epoch 9, with
435 correct predictions among the 521 development selection images. Four
subsequent epochs failed to improve selection accuracy. Training took 2 hours,
3 minutes and 21 seconds. Peak sampled process RSS was 1.72 GB in the four-CPU,
4 GiB Docker worker. [Locked protocol](APTOS_384_V1_PROTOCOL.md).

| Configuration | Accuracy | Macro-F1 | Quadratic kappa | Grade 3 recall | Grade 4 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen features | 78.69% | 0.603 | 0.844 | 6/26 | 22/40 |
| Prior 224-pixel primary | 80.81% | 0.652 | 0.867 | 7/26 | 27/40 |
| 384-pixel primary | 83.49% | 0.697 | 0.907 | 8/26 | 28/40 |

Compared with the prior 224-pixel configuration, the accuracy difference was
+2.69 percentage points, with a paired 95% interval of −0.57 to +5.77 points.
The macro-F1 difference was +0.0453, with an interval of −0.0187 to +0.1112.
Both intervals include zero. Grade 3 recall remained low at 30.77%, and
referable AUROC moved from 0.9813 to 0.9747. The configuration also changes
initialization, augmentation, BatchNorm and optimization, so a matched training
comparison is needed to estimate the effect of resolution. The intervals use
2,000 paired fingerprint-component draws on the reused development selection
cohort. [Training and recall figure](../../reports/aptos_384_v1/training_comparison.png).

Temperature fitting on 262 images yielded `T = 1.560740`. On selection images,
negative log-likelihood moved from 0.5645 to 0.4831 and 15-bin ECE from 0.0868
to 0.0389. The conformal thresholds use 254 representatives from separate
calibration groups. At alpha 0.10, LAC raw coverage was 91.36%, with grade 3
inclusion of 14/26. Deterministic ordinal raw coverage was 85.99% with 54 empty
sets and grade 4 inclusion of 18/40. The five randomized ordinal settings gave
90.21–93.47% raw coverage, average sizes of 1.59–1.74, and grade 3 inclusion of
23–24/26. The report includes raw and fallback results for both alpha values,
all selection images and their 507 fingerprint representatives. These estimates
describe development inclusion using fingerprint groups and unknown patient
relationships. [Full evaluation](../../reports/aptos_384_v1/evaluation.json).

The audit rehashed all 3,163 development images, verified both disk caches,
reconstructed the initial ImageNet tensors and checked the 13 immutable
epochs. It recovered the selection rule from 6,773 saved score rows and
verified 529 BatchNorm updates per layer per epoch. The runner reproduced
the selected epoch's logits exactly. The independent audit reconstructed
14,588 raw and fallback sets and four paired intervals, and verified all 340
preserved artifacts. [Independent audit](../../reports/aptos_384_v1/independent_audit.json).

All 627 local tests passed, including the real-image integration test. GitHub
CI passed on the locked execution source `a446d647535d629f03f46ea2b43a73e29523fff7`.
The rebuilt Docker app passed real-image inference, conformal, simulator and
overload checks. The experiment explorer verified all four alpha and population
views through the running API. The direct-resize checkpoint is available to
the research scorer. [Verification](../../reports/aptos_384_v1/verification.json)
and [explorer evidence](../../reports/aptos_384_v1/docker_registry.json).
