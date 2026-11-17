# Controlled model-input degradation — declared 22 October 2026

This protocol is fixed before generating condition-specific scores. It implements
the controlled synthetic-degradation portion of EXP-08 in MASTER_SPEC_v1.0.
It is a development stress study, not independent confirmation, external-domain
validation, camera simulation, or evidence of clinical safety.

## Frozen population and artifacts

Use all 168 selection images from 42 patients in the original development
manifest order. The complete manifest must have SHA256
`45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d`.
Its only allowed partitions are train/validation/calibration; only validation is
scored. Verify every selected image's actual bytes against its recorded SHA256.
Never load the consumed confirmation population or the official test/IDRiD data.

Use the selected MobileNet DR and quality artifacts, unchanged base calibration,
and independent-temperature conformal sidecar `conformal_r1.json`. Bind their
actual SHA256 values in the output. The DR checkpoint is
`f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527`,
quality checkpoint
`a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43`,
base calibration
`674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737`,
and conformal sidecar
`15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221`.
Do not fit parameters, choose new thresholds, select conditions by their results,
replace models, or change the vacuous ordinal sets after inspecting this study.

## Exactly one geometric preprocessing operation

Decode each original image as RGB and apply the existing black-border crop,
bilinear square letterbox/resize to 224 pixels exactly once. That canonical RGB
array defines the clean control. Apply each perturbation independently to this
same control. Normalize directly with the checkpoint's original mean/std and
run the loaded models in evaluation/inference mode. Do not feed these crops back
through `predict_images`: its border cropping would change geometry after a
brightness/blur perturbation. This research-only path leaves production inference
unchanged. Verify clean scores against ordinary first-pass engine predictions
for every image; require maximum absolute probability error <= 1e-6.

All seven conditions are deterministic and single-pass; no TTA or compute-policy
selection is performed:

| Condition | Operation on canonical RGB |
| --- | --- |
| clean | Identity copy |
| blur_1 | Pillow GaussianBlur radius 1.0 input pixels |
| blur_2 | Pillow GaussianBlur radius 2.0 input pixels |
| exposure_0_5 | Multiply float32 RGB by 0.5, clip [0,255], round with NumPy rint, cast uint8 |
| exposure_1_5 | Multiply float32 RGB by 1.5, same clipping/rounding |
| jpeg_50 | Pillow JPEG quality 50, subsampling 2, optimize=False, progressive=False; decode RGB |
| jpeg_20 | Pillow JPEG quality 20, otherwise identical |

No source image is modified. Package versions and the canonical/perturbed RGB
array hashes are retained with private per-image outputs. Public artifacts contain
aggregates and plots only. JPEG results are bound to the recorded Pillow/libjpeg
implementation. These input-scale manipulations do not model all physical image
acquisition defects; compression after 224px resize differs from source-camera
compression.

## Prespecified analysis

Report complete five-grade metrics, confusion matrices, referable sensitivity,
specificity and false negatives at the unchanged risk threshold 0.5, ECE/Brier/NLL,
mean entropy and mean predicted poor-quality probability for each condition.
Evaluate fixed first-pass LAC and ordinal sets at alpha .10 without recalibration;
include image/per-grade/all-views patient coverage and set sizes. No coverage
guarantee is asserted under corruption. A full five-grade set is uninformative
even when it covers every label.

For each non-clean condition, report paired changes from the same clean images
in DR accuracy, referable sensitivity, NLL, entropy, poor-quality probability,
LAC image coverage and LAC mean size. Use 1,000 paired patient-cluster bootstrap
resamples with replacement, seed 20261022, percentile 2.5/97.5 bounds. Preserve
all views of each sampled patient. Referable sensitivity uses the fixed subset
with original grades >= 2; report its actual patients/images. These intervals
are descriptive and unadjusted across conditions/metrics; no significance or
superiority claim is planned. No monotonicity is assumed.

Report separate original natural-quality (poor/good) by original DR-grade strata,
including count, accuracy, mean entropy and poor-quality probability. Retain DR
labels as underlying disease labels under synthetic corruption, while recognizing
that severe corruption may remove visible diagnostic evidence. Do not reuse
the original natural-quality/defect labels as new ground truth for a corrupted
image. Only clean quality-classification accuracy is meaningful against those
annotations; changed quality scores describe model response, not detection
accuracy on newly labeled defects.

The current ordinal gate requires review on every existing selection image.
Report guarded v2 pointwise action counts under unchanged defaults per condition,
with no workload-capacity or missed-case benefit inferred from all-review output.
This experiment does not independently tune competing policies or establish the
primary dual-budget novelty. Those studies and fresh external evaluation remain
separate tasks.

## Execution and preservation

Implement the research helper and runner with fixture tests for transform,
normalization and paired-cluster calculations. Record protocol/source/input/model
hashes, package versions, exact count and elapsed duration. Private caches live
under ignored `artifacts/robustness_v1`; aggregates under
`reports/robustness_v1`. Refuse existing outputs, so a new execution uses a new
directory. Retain unfavorable or non-monotonic outcomes. Verify clean equivalence
before accepting the experiment and run relevant software/real-model checks.
