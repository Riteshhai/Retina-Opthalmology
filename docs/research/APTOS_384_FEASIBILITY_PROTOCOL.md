# APTOS 384-pixel CPU feasibility probe — 6 November 2026

This bounded engineering probe is declared before its new measurements. The user
requested continued project work while the selected mirror downloads. Source
APTOS development outcomes and prior external outcomes are already known. No
fresh validation, resolution benefit or full paper-training reproduction is claimed.

Use only 32 deterministic representative fitting images from the original frozen
2,115-image APTOS fitting role. Order candidates by SHA256 of
aptos384-feasibility-v1:image_id and keep one representative per known fingerprint
component. Verify source/runtime manifest, source report and image byte hashes.
No selection/calibration/reserved test or external pixel enters the probe.

Use the genuinely trained source EfficientNet-B0 checkpoint from the primary
APTOS optimization study (training seed2026/selected epoch10). This is a temporary
copy for compute feasibility, not the paper's ImageNet-only initialization. Start
each size from the same original checkpoint in a separate fresh process. Compare
direct bilinear RGB resize to224 and384 pixels, ImageNet normalization, FP32,
channels-last, two CPU threads and batch8. No fundus crop or random augmentation.

Each worker performs four actual weighted-cross-entropy AdamW updates on these
32 fitting rows, learning rate3e-4/weight decay1e-4, with ordinary training-mode
BatchNorm. Compute inverse-frequency class weights from the full fitting role.
Check finite losses/gradients and changed temporary parameters. Save no updated
weights or per-image predictions; verify the original checkpoint bytes unchanged.
These eight throwaway updates do not constitute a trained candidate or convergence.

Record preparation/model-step wall time and child-process resident memory. Sample
process-tree RSS/Windows peak working set every20ms, including Windows venv launcher
descendants and verifying the actual worker is observed; describe the method explicitly.
Use fresh processes, size order224 then384 and fixed seed2026, with no warmup claim
or controlled speed comparison. A dataset download and other workstation workloads
may run concurrently. Do not extrapolate a precise full training duration from four
steps, compare losses as accuracy, or make CPU speed-superiority claims.

Require at least2 GB currently available memory before spawning each worker.
Fail visibly on source, split, integrity, finite-state or memory errors. Preserve
all original datasets/master/checkpoints/reports and all prior results. Record
source hashes/commit, private selected-input digest and actual package versions.
Fixture tests must reject held-out/mixed roles and validate deterministic geometry.
Commit the protocol and tested runner before actual probe execution. Actual
measurements, independent verification and results notes follow successful runs.
Formal manuscript drafting and full384px paper training remain separate work.
