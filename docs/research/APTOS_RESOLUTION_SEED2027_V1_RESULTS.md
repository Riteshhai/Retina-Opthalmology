# APTOS input resolution across two training seeds

The seed-2027 repeat used the same EfficientNet-B0 recipe, ImageNet weights
and frozen APTOS development roles as the seed-2026 comparison. The 224-pixel
arm stopped after 10 epochs and 2,650 updates, selecting epoch 6. The 384-pixel
arm stopped after 9 epochs and 2,385 updates, selecting epoch 5. Both stopped
under the same four-epoch patience rule.
[Locked protocol](APTOS_RESOLUTION_SEED2027_V1_PROTOCOL.md).

| Seed | Contrast | Pixels | Epoch | Accuracy | Macro-F1 | Quadratic kappa | Grade 3 recall | Grade 4 recall | Referable AUROC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | Selected | 224 | 12 | 83.11% | 0.715 | 0.908 | 15/26 | 29/40 | 0.9793 |
| 2026 | Selected | 384 | 9 | 83.49% | 0.697 | 0.907 | 8/26 | 28/40 | 0.9747 |
| 2026 | Equal updates | 224 | 5 | 78.31% | 0.651 | 0.869 | 21/26 | 21/40 | 0.9769 |
| 2026 | Equal updates | 384 | 5 | 80.42% | 0.679 | 0.896 | 17/26 | 28/40 | 0.9804 |
| 2027 | Selected | 224 | 6 | 81.38% | 0.693 | 0.894 | 15/26 | 25/40 | 0.9806 |
| 2027 | Selected | 384 | 5 | 82.53% | 0.681 | 0.893 | 12/26 | 24/40 | 0.9805 |
| 2027 | Equal updates | 224 | 5 | 80.81% | 0.653 | 0.879 | 12/26 | 21/40 | 0.9751 |
| 2027 | Equal updates | 384 | 5 | 82.53% | 0.681 | 0.893 | 12/26 | 24/40 | 0.9805 |

| Seed | Contrast, 384 minus 224 | Accuracy difference in percentage points | Macro-F1 difference |
| ---: | --- | ---: | ---: |
| 2026 | Selected checkpoints | +0.38 [−2.32, +3.10] | −0.0179 [−0.0726, +0.0351] |
| 2026 | Epoch 5 | +2.11 [−0.95, +5.17] | +0.0283 [−0.0181, +0.0752] |
| 2027 | Selected checkpoints | +1.15 [−1.92, +4.26] | −0.0123 [−0.0658, +0.0417] |
| 2027 | Epoch 5 | +1.73 [−0.96, +4.41] | +0.0275 [−0.0179, +0.0744] |

At both seeds, 384 pixels increased selected-checkpoint accuracy by 0.38 to
1.15 percentage points, while selected macro-F1 fell by 0.0123 to 0.0179.
At epoch 5 after 1,325 updates per arm, accuracy increased by 1.73 to 2.11
points and macro-F1 increased by 0.0275 to 0.0283. Selected grade 3 recall
was lower at 384 pixels in both seeds.

All eight paired 95% intervals include zero. Each uses 2,000 paired draws
over 507 fingerprint components with fixed checkpoints. Both seeds used the
same 521 images for checkpoint selection and comparison. Patient identities
were unavailable, and the range across two seeds is descriptive. The repeat
used fitting and selection images for model scoring. Calibration, the
reserved APTOS test and external cohorts supplied no repeat outcomes.
[Aggregate evaluation](../../reports/aptos_resolution_seed2027_v1/evaluation.json)
and [paired interval figure](../../reports/aptos_resolution_seed2027_v1/paired_resolution_seeds.png).

The original 384-pixel worker exited with an I/O error after saving eight
complete checkpoint and score pairs. The replacement resumed from epoch 8
with the same source commit, image, inputs and resource limits. The trainer
restored optimizer, scheduler and RNG state from the validated checkpoint
chain. It accumulated 4,770.47 training seconds across completed work,
excluding the failed partial epoch and restart gap. The 224-pixel worker
recorded 1,917.79 training seconds. Peak sampled process RSS was 1.78 GB
at 384 pixels and 1.06 GB at 224 pixels.

The independent audit rehashed 2,636 fitting and selection images and 352
preserved artifacts. It verified 19 immutable checkpoints and 9,899 saved
selection rows, matching RNG states at nine common epochs, and all eight
paired intervals. Replay of both selected checkpoints in the pinned Linux
image matched the saved logits exactly.
[Independent audit](../../reports/aptos_resolution_seed2027_v1/independent_audit.json).

The rebuilt Docker services matched all 73 runtime files and passed genuine
DeepDRiD HTTP prediction, conformal and simulator checks. Streamlit AppTest
matched all eight checkpoint rows to the live API. The locked source passed
634 local tests and source CI, and 39 targeted publication checks passed
locally.
[Deployment and recovery verification](../../reports/aptos_resolution_seed2027_v1/verification.json)
and [explorer verification](../../reports/aptos_resolution_seed2027_v1/docker_registry.json).
