# APTOS 384-pixel development protocol

Locked on 9 November 2026 under `MASTER_SPEC_v1.0`. This experiment measures one
fixed ImageNet-initialized EfficientNet-B0 configuration on the existing APTOS
development cohorts. The primary training seed is 2026.

Use the frozen 2,115 fitting, 521 selection and 527 calibration images. Reuse
the audited fingerprint components and the existing calibration-role assignment.
Verify the development manifest, source annotations, original image hashes and
all comparator score hashes. Access the reserved internal-test manifest only for
the identity and split audit. All fitting and scoring use development images.

Initialize the torchvision backbone from `EfficientNet_B0_Weights.IMAGENET1K_V1`
with full SHA-256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`.
Construct a fresh five-way linear head after seeding Torch. Resize RGB directly
to 384 by 384 with bilinear interpolation, then use ImageNet normalization.
During fitting apply independent horizontal and vertical flips at probability
0.5, nearest-neighbor rotation uniformly sampled from −15 to 15 degrees with
black fill, and brightness, contrast and saturation factors sampled from 0.9 to
1.1 in random order. Selection and calibration use deterministic preprocessing.

Train all parameters with active BatchNorm, weighted cross-entropy with
`N / (5 * n_c)` class weights, and AdamW at learning rate 0.0003 and weight decay
0.0001. Use cosine annealing to zero over 15 epochs. Four-image microbatches
accumulate gradients within each eight-image optimizer window. Normalize every
microbatch loss sum by the weight sum of the whole window, including the final
short window. BatchNorm and stochastic layers see the microbatch. Use CPU FP32,
four Torch threads and channels-last tensors. The split, torchvision model and
CPU microbatch scheme differ from the published CUDA training setup.
[Base paper, section 3.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC13512306/#sec3dot3-diagnostics-16-02645)
and [torchvision weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).

Select by strictly higher accuracy on the 521 selection images. The first epoch
wins ties. Stop after four consecutive epochs without improvement or 15 epochs.
Stream the resized RGB cache from disk. Require 2 GiB of available RAM and 5 GiB
of available disk at launch. Check the 3-hour training budget at epoch boundaries.
After each update check available RAM against 256 MiB and process RSS against
4 GiB. A resource interruption leaves the run incomplete and preserves completed
epochs. Resume requires the same source, inputs, model initialization and epoch
chain. Save the epoch weights, optimizer, scheduler and random states together
with a hash of the corresponding selection scores.

Compare the selected 384-pixel checkpoint with the frozen-feature EfficientNet-B0
baseline and the fixed primary 224-pixel optimization run. Report raw accuracy,
macro-F1, quadratic weighted kappa and per-grade recall. Use 2,000 paired
fingerprint-component bootstrap draws with seed 20261109 for accuracy and
macro-F1 intervals. The comparison estimates the joint configuration change
because initialization, preprocessing, BatchNorm and optimization also differ.

After selection, fit temperature on the fixed 262 temperature-role images and
fit conformal thresholds on the separate 254 representatives. Report LAC,
deterministic ordinal and all five declared randomized ordinal settings at
alpha 0.10 and 0.05. Preserve raw and argmax-fallback results for all 521
selection images and their 507 fingerprint representatives. Report empirical
development coverage with the limits of fingerprint grouping and reused cohorts.

Run the input check with `python scripts/run_aptos_384.py`. Execution uses
`python scripts/run_aptos_384.py --execute --source-commit <commit>` from the
clean commit that contains this protocol. Add `--resume` to resume a matching
run in the same environment. Build the worker with
`docker build --target aptos-training -t qsart:aptos-training .`. On Windows,
`scripts/run_aptos_384_docker.ps1` checks the inputs and accepts `-Execute` to
start the run. Its container has four CPUs, a 4 GiB memory limit and no network.
It mounts the checkout and APTOS source read-only, with separate writable mounts
for this experiment's outputs. Keep weights, pixels and individual predictions
local. Publish aggregate evidence under `reports/aptos_384_v1`. The direct-resize
checkpoint is eligible for the research scoring path only.
