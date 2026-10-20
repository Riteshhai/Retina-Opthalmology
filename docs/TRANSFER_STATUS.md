# Development v1 execution — 20 October 2026

The interrupted implementation was recovered and completed locally. Existing
pilot data, weights and reports were preserved. The new predeclared protocol is
docs/research/DEVELOPMENT_V1_PROTOCOL.md; it precedes new confirmation predictions.

## Data and learning

- Development: 1,184 verified images / 296 official-training patients. All four
  ambiguous subjects were excluded. Actual partitions are 848 fitting images /
  212 patients, 168 selection images / 42 patients, and 168 calibration images /
  42 patients. Requested proportions were rounded within severity strata.
- Development CSV SHA256:
  45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d.
- Reserved confirmation: 340 images / 85 official-validation patients, excluding
  all 15 inspected pilot holdout patients. This is internal same-domain data.
  Its CSV SHA256 is
  21cfd7aa43660a236903430207b27cc3a8366a9129cbcbfa513eee0666baa9b7.
- One-time official torchvision ImageNet weights were downloaded and hashed.
  The pretrained backbone and BatchNorm statistics remained unchanged. New heads
  learned from cached original and flipped fitting views at 224-pixel resolution.
- MobileNet DR: early-stopped after 30 epochs; selected epoch 15, selection
  accuracy 0.63095, macro-F1 0.53724, unweighted NLL 1.19712.
- ResNet18 DR: early-stopped after 35 epochs; selected epoch 20, selection
  accuracy 0.61310, macro-F1 0.51601, unweighted NLL 1.21230.
- The declared macro-F1/NLL rule selects MobileNet. These are selection results,
  not an unbiased improvement estimate over the differently sampled old pilot.
- Quality MobileNet: early-stopped after 18 epochs; selected epoch 3, overall
  selection accuracy 0.74405 and macro-F1 0.73726. Genuine mapped clarity,
  field-definition and artifact heads were trained as well; their accuracy is
  lower and must be reported separately, without implying reliable defect detection.
- Fixed statistical quality comparator: selection accuracy 0.72619, macro-F1
  0.72301, versus training-prior baseline accuracy 0.58333. Fitting standardization
  and regression used training data only.

## Verification and boundaries

Fixed an unfinished source-split check (`training` versus actual `official_train`).
Added tests for cohort exclusion, connected fingerprint groups, natural-code
mapping, frozen-backbone caches, real optimizer steps, serialization, changed
pixels, confirmation rejection and patient-cluster reporting. Tests use generated
fixtures only; the metrics above come from actual retinal images.

Source was committed before training; checkpoints additionally hash the exact
training source, pretrained weights, input bytes and feature-cache metadata.
Calibration and confirmation execution statuses will be appended from artifacts.
No confirmation results have been used to choose or tune these models.
