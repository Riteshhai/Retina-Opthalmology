# Model development v1 — fixed before new model results

Recorded 20 October 2026, before executing the new transfer-learning experiments.
This implements the approach agreed on 18 October before work was interrupted.
The old 60-image pilot holdout is inspected development evidence and will not
be used to select, tune or claim final performance of the new models.

## Cohorts and access boundaries

Use all official DeepDRiD training patients except the four subjects containing
ambiguous eye/view annotations: expected 296 patients / 1,184 images. Assign
patient-separated fitting, selection-validation and calibration groups using
seed 2026 and requested 70/15/15 proportions, stratified by maximum verified eye
grade. Fingerprint-connected subjects stay in the same partition; report actual
rounded group sizes. Keep original clinical identifiers and defect codes intact.

Reserve all official-validation patients except the 15 inspected pilot holdout
patients: expected 85 patients / 340 images. This is fresh internal same-domain
confirmation, not independent external-domain validation. Source annotations were
already available; no new model outputs on these patients have been inspected.
Audit cross-cohort subject, exact-image and perceptual overlap before evaluation.
Freeze model, calibration and policy artifacts before opening confirmation results.

Some old pilot fitting patients can now be in selection/calibration. Therefore
start the new models only from external ImageNet weights, never the pilot weights.

## Fixed training and selection rules

Compare ImageNet-pretrained MobileNetV3-small and ResNet18 for DR, with 224-pixel
fundus crop/pad/RGB normalization shared with inference. Freeze the backbone,
including BatchNorm statistics. Cache original and horizontal-flip fitting views;
use original selection views. Optimize new linear heads with AdamW, learning
rate 0.01, weight decay 0.0001, head batch 64, maximum 100 epochs, patience 15,
class weights from fitting labels, seed 2026. Extract with batch 16, two CPU threads
and two loader workers. Reduce learning rate on selection NLL plateaus.

Select each checkpoint, then the DR candidate, lexicographically by highest
selection five-class macro-F1, followed by lowest unweighted NLL. Do not select
on accuracy alone or on the confirmation cohort. Train binary overall quality
and mapped clarity/field-definition/artifact heads on MobileNet features. Select
that checkpoint by overall-quality macro-F1, then overall-quality unweighted NLL.
Detailed defect labels do not drive checkpoint selection.

Report a training-class-prior DR/quality baseline and a fixed classical quality
baseline: fitting-only standardized image statistics and class-balanced logistic
regression, C=1, maximum 1,000 iterations, seed 2026. Its 128-pixel statistics are
RGB means, green contrast/quantiles, Laplacian variance, gradient energy, exposure
fractions and original dark-border fraction. No feature or threshold search on
confirmation labels is allowed. This comparator need not outperform deep features.

Fit first-pass and flip-TTA temperatures only on the dedicated calibration group
after candidate selection. Report NLL/ECE/Brier both before and after, including
worse outcomes. Preserve artifact hashes, actual input bytes, feature extraction
configuration, weight URL/hash and exact source hashes. Calibration-fit metrics
are distinct from selection or confirmation performance.

The fixed exploratory triage costs and budgets from the prior implementation
remain unchanged for this round. No favorable operating point will be selected
using confirmation labels. Record infeasible budgets and unresolved recapture
cases. Model improvements do not by themselves validate the triage contribution.

## Artifact and completion rules

Store new weights under checkpoints/development_v1 and preserve the running pilot.
Only switch the application after the candidate is selected and calibrated and
real model-backed integration checks pass. Keep images, identifiers, feature
caches and per-image predictions out of Git; publish aggregate evidence only.
Use frozen source and artifact checksums to distinguish training runs. Never
substitute generated predictions for a failed run.

## Pretrained source record

Use the installed torchvision IMAGENET1K_V1 enums for the two candidate models;
download URLs and complete file hashes are captured in
reports/development_v1/pretrained_sources.json. The official model definitions
are documented for [MobileNetV3-small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
and [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
Torchvision code is BSD-3-Clause. Its [pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license)
distinguishes model/dataset terms from the software license. Do not assert blanket
commercial rights or redistribute these weights under the code license. This is
a local academic research experiment; downloaded weights remain outside Git.
