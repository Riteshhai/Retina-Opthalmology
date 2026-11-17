# APTOS matched resolution repeat protocol

Repeat the 224- and 384-pixel EfficientNet-B0 arms with seed 2027, the next
integer after the published seed 2026. Both seed-2026 results were available
before this protocol. The new seed and both contrasts are fixed before fitting.
[Original matched protocol](APTOS_MATCHED_RESOLUTION_V1_PROTOCOL.md).

Use the audited ImageNet weights, the same five-way head initializer with the
new seed, and the exact trainer bytes from the first matched pair. Keep direct
RGB resize, augmentations, shuffled fitting order within each seed, active
BatchNorm, four-image FP32 microbatches, eight-image optimizer windows, weighted
AdamW, the 15-epoch cosine schedule, strict accuracy selection with earliest
ties, and patience four. Both arms use the same 2,115 fitting and 521 selection
images. Verify the original image bytes and reuse each audited, read-only pixel
cache. The frozen calibration and test partition bindings are checked for
integrity. Model forwards use fitting and selection roles exclusively. This
repeat measures grading sensitivity to training randomness. The prior
temperature and conformal results are preserved with their original seed.

Use the same pinned nonroot Linux Docker image with four CPUs, 4 GiB, no network,
and read-only source and input mounts. Run the two arms sequentially in the
order 224, then 384, to limit workstation contention. Preview each arm with
`./scripts/run_aptos_384_docker.ps1 -Seed 2027 -ImageSize 224` or `-ImageSize 384`.
Use `-Execute` from the clean committed source. Add `-Resume` only for an
interrupted arm with matching source, lock and immutable epoch chain.

For each seed, report 384 minus 224 for the two checkpoints selected by the
shared rule and for both arms at epoch 5 after 1,325 updates. Use the same 2,000
paired fingerprint-component draws with seed 20261109 for accuracy and macro-F1.
Retain per-grade recall, quadratic weighted kappa and referable AUROC. Show both
seed pairs and the descriptive range of their differences. The intervals hold
the checkpoints fixed and use the same 521 development selection images.
Patient identities are unavailable. Two seeds do not estimate the distribution
of resolution effects over training runs.

Audit both immutable checkpoint chains, saved score rows, exact selected replay
and seed-2027 RNG states at common epochs. Reconstruct the paired differences
and retain the byte-pinned seed-2026 results. Private arm reports and scores stay
under the ignored artifact and checkpoint directories. Publish the completed
two-seed aggregate under `reports/aptos_resolution_seed2027_v1`. The production
checkpoint is outside this study.
