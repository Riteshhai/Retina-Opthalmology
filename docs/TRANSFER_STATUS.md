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
Calibration and reserved confirmation have now completed; details follow.
No confirmation results have been used to choose or tune these models.


## Completed calibration and confirmation

`finalize_development.py` selected MobileNet, copied selected artifacts, fitted
original/flip temperatures on 168 calibration images and froze the selection lock.
Calibration-fit DR temperature 1.46215 reduced NLL 1.13744 → 1.08790 and ECE
0.12982 → 0.06202. These are fitting diagnostics only.

`evaluate_confirmation.py` then validated all frozen hashes and predicted all
340 images / 85 patients. DR accuracy is 60.00%, macro-F1 0.53176, QWK 0.72650,
referable AUROC 0.92482 and calibrated ECE 0.03999. The patient-cluster accuracy
interval is 52.65–67.06%. Quality accuracy is 74.12%; classical quality is 73.24%.
Natural defect accuracies are 49.41% clarity, 48.53% field definition and 35.59%
artifact. Artifact accuracy falls below majority-class accuracy. All 63 fixed
policy/scenario rows and 27 infeasibilities are retained. No confirmation tuning
occurred. [Full results and limits](research/CONFIRMATION_V1_RESULTS.md) describe
exploratory matched-workload comparisons and the consumed-evaluation boundary.

Selected artifacts remain local in `checkpoints/development_v1/selected`:

- DR SHA256: f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527.
- Quality SHA256: a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43.
- Calibration SHA256: 674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737.

`package_confirmation.py` creates a checkpoint/calibration-bound public model card
and all-grid workload plot from frozen aggregate reports, without fitting anything.
The API refuses to show recorded metrics for mismatched weights or calibration.
Local `.env` now selects v1 container paths; original pilot checkpoints remain.

## Current deployment and verification

- Full real-model opt-in suite: 89 passed, 39.07 seconds, one upstream deprecation
  warning. `python -m ruff check .` passed.
- Rebuilt Docker CPU image; both services healthy. Actual new-model JPEG single,
  batch, second-pass, unsupported-upload, readiness, docs and dashboard checks
  passed. Model card matches the loaded weights/calibration.
- All 31 container runtime Python hashes match host; actual dependencies captured.
- Browser displays current model/version, 60.0% DR and 74.1% quality results with
  internal-scope text. Screenshot: reports/dashboard_v1_preview.png.
- CPU strategy benchmark: 48-image deterministic selection prefix, batch/cohort
  chunk 8, model forward 1, 3 repeats, 2 CPU threads, 224px. Training/confirmation
  inference had finished. Ordinary workstation/report activity remained. The
  sequential timing order does not support a speed-advantage claim; all strategy
  accuracy/workload comparisons and timing notes are in reports/development_v1.

Reproduction commands for the new CPU evidence and deployment check:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_cpu.py --dr-checkpoint checkpoints/development_v1/selected/dr.pt --quality-checkpoint checkpoints/development_v1/selected/quality.pt --calibration checkpoints/development_v1/selected/calibration.json --manifest data/processed/deepdrid/development_v1.csv --split validation --output artifacts/development_v1/cpu_benchmark.json --max-images 48 --batch-size 8 --repeats 3
.\.venv\Scripts\python.exe scripts/verify_deployment.py --manifest data/processed/deepdrid/confirmation_v1.csv --api http://127.0.0.1:18000 --dashboard http://127.0.0.1:18501 --output reports/deployment_v1.json
```
