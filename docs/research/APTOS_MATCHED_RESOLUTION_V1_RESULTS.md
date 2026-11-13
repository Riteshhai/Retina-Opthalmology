# Matched APTOS input resolution results

The 224-pixel EfficientNet-B0 arm completed all 15 epochs and 3,975 optimizer
updates on 9 November 2026. The strict accuracy rule selected epoch 12. The
384-pixel reference selected epoch 9 and stopped at epoch 13 after 3,445 updates.
Both used seed 2026, the same ImageNet starting weights and training recipe,
and the frozen APTOS development roles. The reference results were available
when the 224-pixel protocol was locked. [Protocol](APTOS_MATCHED_RESOLUTION_V1_PROTOCOL.md).

| Contrast | Input | Epoch | Accuracy | Macro-F1 | Quadratic kappa | Grade 3 recall | Grade 4 recall | Referable AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected | 224 | 12 | 83.11% | 0.715 | 0.908 | 15/26 | 29/40 | 0.9793 |
| Selected | 384 | 9 | 83.49% | 0.697 | 0.907 | 8/26 | 28/40 | 0.9747 |
| Equal updates | 224 | 5 | 78.31% | 0.651 | 0.869 | 21/26 | 21/40 | 0.9769 |
| Equal updates | 384 | 5 | 80.42% | 0.679 | 0.896 | 17/26 | 28/40 | 0.9804 |

The selected checkpoints compare the outcomes of the shared selection rule.
Epoch 5 compares both resolutions after 1,325 updates, before either run can
stop under the patience rule. The direction of the macro-F1 difference changes
between the two contrasts. All four paired 95% intervals include zero.

| Contrast, 384 minus 224 | Accuracy difference in percentage points | Macro-F1 difference |
| --- | ---: | ---: |
| Selected checkpoints | +0.38 [−2.32, +3.10] | −0.0179 [−0.0726, +0.0351] |
| Epoch 5 | +2.11 [−0.95, +5.17] | +0.0283 [−0.0181, +0.0752] |

The 2,000 paired draws resample 507 fingerprint components covering 521 reused
development selection images, with fixed predictions. The intervals describe
one training seed on these images. Patient identity and between-seed variation
remain unknown. [Comparison figure](../../reports/aptos_224_matched_v1/resolution_comparison.png).

Worker training time was 2,828.64 seconds at 224 pixels and 7,400.51 seconds at
384 pixels. Peak sampled process RSS was 1.07 GB and 1.72 GB, respectively. The
measurements came from separate four-CPU runs with the same 4 GiB container
limit. The Windows host entered standby once during the 224-pixel run. The same
worker continued after wake, and the reported training time comes from its
elapsed-time counter. A temporary system execution request was released at exit.

The 224-pixel temperature fit on 262 images gave `T = 1.515979`. Selection NLL
decreased from 0.5983 to 0.5181 and 15-bin ECE from 0.0972 to 0.0518. Conformal
thresholds used 254 representatives from separate calibration groups. At alpha
0.10, raw LAC coverage was 91.55% with 21/26 grade 3 inclusion. The deterministic
ordinal sets gave 90.21% raw coverage, 38 empty sets and 26/40 grade 4 inclusion.
The five randomized ordinal settings ranged from 85.80% to 92.32% raw coverage
with 11–31 empty sets. The report retains all 14 settings with raw and fallback
views for all 521 selection images and the 507 representatives. These are
empirical development inclusion rates. [Evaluation](../../reports/aptos_224_matched_v1/evaluation.json).

The audit rehashed 3,163 development images and 350 preserved artifacts. It
checked 15 immutable epochs, 7,815 saved selection rows, 14,588 raw and fallback
sets, and all four intervals. The selected checkpoint replay matched the saved
logits exactly. The auditor reconstructed the seeded head inside the pinned
Linux image and verified matching shuffle, augmentation and Torch RNG states
at all 13 common epochs. Study scoring used the development roles exclusively.
[Independent audit](../../reports/aptos_224_matched_v1/independent_audit.json).

All 631 local tests passed, including genuine-image inference. GitHub CI passed
on execution source `cc4f4450cb9b2663def6cb1e637949cdfdd50782`. Both rebuilt Docker
services were healthy and matched all 73 runtime source and configuration files.
The deployed DeepDRiD checkpoint, model card and conformal sidecar passed actual
HTTP prediction, batch and simulator checks. Streamlit AppTest exercised the four
checkpoint rows and all four conformal views against the live API. The new
checkpoint is available to the research scorer. [Deployment verification](../../reports/aptos_224_matched_v1/verification.json)
and [explorer verification](../../reports/aptos_224_matched_v1/docker_registry.json).
