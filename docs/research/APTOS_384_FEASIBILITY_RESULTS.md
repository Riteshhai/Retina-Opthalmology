# APTOS 384-pixel CPU feasibility — 6 November 2026

**VERIFIED COMPLETE for the bounded engineering probe.** Full paper training and
scientific acceptance remain IN PROGRESS. Executed from clean
`c1e52fd21f43e38de78007e1376055276be5865d` after the
[declared protocol](APTOS_384_FEASIBILITY_PROTOCOL.md).

Both image sizes completed four genuine weighted-cross-entropy AdamW updates
on the same 32 deterministic source-fitting representatives. Each size started
in a fresh process from the same genuinely trained EfficientNet-B0 checkpoint.
Temporary parameters changed and all losses/gradients remained finite. The eight
throwaway updates produced no candidate checkpoint or saved predictions.
Production/source model files remain unchanged; no selection, calibration,
reserved-test or external image was used.

| Direct RGB resize | Fitting images | Updates / batch | Pixel preparation seconds | Four update seconds | Sampled process-tree RSS (decimal GB) |
| --- | ---: | --- | ---: | ---: | ---: |
| 224 | 32 | 4 / 8 | 2.98046 | 2.71586 | 1.310 |
| 384 | 32 | 4 / 8 | 3.00417 | 8.24752 | 2.742 |

The 384-pixel path was executable on this workstation. Its observed process-tree
RSS was about 2.74 GB. This is a short feasibility observation, not a full-training
RAM requirement or controlled speed comparison. The two-GB availability guard
was a pre-run floor; it is not an upper bound on usage. Pixel preparation excludes
Python/model initialization; the four-step time excludes input preparation and
model/optimizer construction. Memory includes the fresh worker and Windows venv
launcher descendants, sampled every 20 ms; Windows peak working set is also saved.
The sampler verified that it observed the actual worker process.

Both probes used FP32, channels-last, two CPU threads, batch eight, direct
bilinear square RGB resize and ImageNet normalization. Class weights came from
the full frozen 2,115-image fitting role. AdamW used learning rate 3e-4 and weight
decay 1e-4, with training-mode BatchNorm. No augmentation or fundus-border crop
was applied. The initialization was an already selected 224-pixel source model,
not ImageNet-only weights, so this does not reproduce the paper's training setup.
No classification/coverage outcome, convergence or training-duration extrapolation
is inferred. The mirror download and other workstation workloads were concurrent.

The separate saved-record audit reconstructs the hash-selected 32 fitting rows,
class weights and all original image hashes; matches both worker records; verifies
the executed source/protocol bytes against Git; and rechecks the original
checkpoint plus 307 preserved artifacts. It does not repeat timings, gradients
or neural fitting. This distinction is recorded in the audit scope.

[Measured feasibility](../../reports/aptos_384_feasibility_v1/feasibility.json) and
[saved-record/input audit](../../reports/aptos_384_feasibility_v1/record_audit.json).
The next training experiment needs its own fixed source initialization, 384-pixel
augmentation/optimization/selection protocol and resource budget. Reserved APTOS
tests and consumed external outcomes remain excluded from tuning. Formal
manuscript drafting remains deferred.
