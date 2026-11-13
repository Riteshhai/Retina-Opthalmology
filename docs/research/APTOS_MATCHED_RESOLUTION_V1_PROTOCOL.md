# Matched APTOS input resolution protocol

Lock the 224-pixel arm against the published 384-pixel run from source
`a446d647535d629f03f46ea2b43a73e29523fff7`. The reference evaluation SHA-256 is
`746e47de7dba071c82e4ea15736daa19021a3a9d804ca42b2351d2554e635daf`.
The reference outcomes were already available when the matched arm was designed.
The historical 224-pixel arm used a different recipe and belongs to the earlier
bundled comparison. [384-pixel protocol](APTOS_384_V1_PROTOCOL.md).

## Matched controls

Use 224×224 bilinear direct RGB resize and the same preprocessing and training
code as the reference. Match the ImageNet EfficientNet-B0 weights, initialized
backbone and BatchNorm buffers, fresh five-way head seed, augmentation, fitting
order, weighted AdamW, active BatchNorm, FP32, four-image microbatches and
eight-image optimizer windows. Match seed 2026, the 15-epoch cosine schedule,
strict improvement in selection accuracy, earliest tie, and patience four.
The trainer byte check permits only the resolution in the checkpoint name to
vary. Bind the same source cohort of 2,115 fitting, 521 selection and 527
calibration images, and verify the runtime CSVs and development image hashes.

Run the exact reference Docker image with four CPUs, 4 GiB memory and no
network. Mount the project and APTOS source read-only, with writable directories
limited to the new study. On Windows, check with
`./scripts/run_aptos_384_docker.ps1 -ImageSize 224`, then execute the clean
committed protocol with `-ImageSize 224 -Execute`. Resume a matching incomplete
run with `-ImageSize 224 -Execute -Resume`.

## Comparisons

Compare the checkpoints chosen by the shared selection rule. Also compare both
arms at epoch 5, the earliest epoch permitted by patience four. This fixed
contrast uses the same 1,325 optimizer updates per arm. Report both contrasts in
the direction 384 minus 224. Retain accuracy, macro-F1, quadratic weighted kappa,
per-grade recall and referable AUROC. Use 2,000 paired fingerprint-component
bootstrap draws with seed 20261109 for accuracy and macro-F1 intervals. The
intervals describe the fixed predictions on reused development images. Training
seed variability and patient identity remain unmeasured.

Fit the 224-pixel temperature on the same 262 temperature-role images and the
conformal thresholds on the same separate 254 representatives. Retain all 14
settings and the raw and fallback views for 521 images and 507 fingerprint
representatives. All scoring uses the development roles. The reserved APTOS
test, official competition test and external datasets remain outside the study.
The new checkpoint is restricted to the research scorer. Compare observed timing
and memory as workstation measurements from separate runs.

Reconstruct epoch metrics, both paired contrasts and all conformal sets from
saved scores, and verify common-epoch shuffle and augmentation RNG states against
the reference. Publish aggregates in `reports/aptos_224_matched_v1`. Store
checkpoints, per-image scores and pixel caches in the ignored directories.
