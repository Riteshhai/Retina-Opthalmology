# Model implementation and measured evidence

This is a research prototype, not a clinically validated diagnostic system.

## Implemented model pathway

- PyTorch MobileNetV3-small and ResNet-18 candidate backbones; five-grade DR head and independently trained overall-quality model. Optional additional quality heads accept genuinely labeled manifest columns.
- Explicit `--pretrained` permits a one-time torchvision ImageNet weights download. Default training initializes weights locally; inference always reconstructs with `weights=None` and never downloads weights.
- Manifest-backed supervision, patient/split overlap checks, reproducible seed, class-balanced cross-entropy, validation-loss checkpoint selection, early stopping, ReduceLROnPlateau scheduling, CPU and optional CUDA training, supported CUDA mixed precision, resume state, CSV histories and loss curves.
- Shared `qsart.preprocessing.preprocess_image` fundus crop, RGB letterbox resizing, and ImageNet mean/std normalization for training, calibration and inference. Random horizontal flips are training augmentation.
- State-dictionary checkpoints loaded on CPU with `weights_only=True`. Missing, untrained, malformed, nonfinite or software-test artifacts fail production loading. Metadata records task, trained steps, candidate architecture, image size, seed, split counts/fingerprints, manifest checksum and code revision.
- Separate calibration partition fits positive temperature scaling for original and horizontal-flip mean-logit predictions. NLL, multiclass Brier score, ECE and reliability diagrams describe **calibration-fit data**, not independent test performance.
- Model-backed inference returns five-grade probabilities, grade >=2 referable probability, quality probabilities and poor-quality probability, calibrated normalized entropy, selected second-pass status and measured image processing time.

The primary implemented second-pass candidate is a horizontal flip averaged at the logits. Candidate selection was an engineering choice for low memory; experimental superiority has not been established. Comparisons support single pass, full second pass, uncertainty trigger, severity trigger and risk/uncertainty-ranked population budget. A batch with N records and budget B permits at most floor(B*N) extra passes; a single record with B<1 gets none. This does not establish a streaming CPU time guarantee. DR and quality each receive the same selected extra image pass.

## Executed software verification

On October 18, 2026, `.venv/Scripts/python.exe -m pytest tests/test_models.py -q` completed **12 passed in 19.65 seconds** using CPU PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu. Ruff checks passed for all model, training, calibration, inference and associated script/test files.

Coverage includes both backbone tensor shapes and probability normalization, CPU checkpoint equality, actual optimizer-updated generated fixtures, production rejection of those software-test artifacts, wrong calibration checksum rejection, temperature-fit loss, population-budget accounting, empty/invalid batches, shared preprocessing, resume, subject overlap rejection and rejection of quarantined/ineligible/unreadable manifest records. Generated test images and their checkpoints live only in pytest temporary directories and carry `data_purpose=software_test`; no research metrics derive from them.

## Genuine retinal pilot

Training and calibration completed on the verified DeepDRiD pilot manifest `data/processed/deepdrid/pilot_manifest.csv`: 160 train images, 20 model-selection validation images, 20 calibration images, and 60 separately preserved official-validation holdout images. Subjects are grouped across views; the official final test set is untouched. Both actual runs used MobileNetV3-small, 96px, three epochs, batch size 8, two CPU threads, seed 42, no pretrained initialization. Each run completed 60 genuine optimizer updates. This deliberately small pilot demonstrates execution and cannot establish useful clinical performance.

DR validation loss selected epoch 1 (20 completed optimizer steps) at loss 1.609593 and accuracy 0.20. Final training accuracy reached 0.45625, while validation accuracy stayed 0.20 for all three epochs. Quality validation loss selected epoch 2 (40 steps) at loss 0.685183 and accuracy 0.70. Final quality training accuracy reached 0.66875; its 0.70 validation accuracy equals the majority-poor baseline (14/20 images), so it is not evidence of predictive skill. Curves, histories and checkpoint metadata are saved locally under `checkpoints/dr/` and `checkpoints/quality/`.

`checkpoints/calibration.json` was fitted on the 20 dedicated calibration images only, using the selected checkpoints. The manifest SHA256 is `e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0`.

The pilot checkpoint's `code_version` records the repository HEAD at training time (`c5c0a3bb5d8db5bd16278aed596170410fa6f24c`), while the newly implemented source was still uncommitted. That revision alone is not an exact source snapshot for these runs. Preserve the delivered source, configuration, package versions and histories with the artifacts; the original locked checkpoint/calibration checksums have not been rewritten after evaluation was authorized.

| Calibration-fit diagnostic | DR before | DR after | Quality before | Quality after |
| --- | ---: | ---: | ---: | ---: |
| NLL | 1.605348 | 1.595836 | 0.701899 | 0.693547 |
| Multiclass Brier | 0.798419 | 0.796205 | 0.508747 | 0.500400 |
| ECE (15 bins) | 0.005203 | 0.032086 | 0.119890 | 0.100995 |

The DR fitted temperature was 0.159717; quality reached the configured maximum temperature 20. **DR ECE worsened despite the optimized NLL decreasing.** These mixed diagnostics are reported as measured; fitting NLL does not guarantee ECE improvement. The small calibration sample contains multiple views per subject and does not support an independent calibration-performance claim. Separate temperatures for horizontal-flip mean logits were also genuinely fitted and stored. Reliability diagrams are next to the calibration JSON.

The quality pilot uses the official overall-quality labels, poor=0 and good=1. Raw natural clarity, field-definition and artifact annotations remain available in the manifest but are not learned by this first binary-quality checkpoint. Ordinal raw defect codes require explicit mapped heads in a future experiment; they must not be silently interpreted as consecutive class indices.

## Actual CPU inference measurement

`artifacts/pilot_cpu_benchmark.json` and its companion prediction CSV record actual artifact-backed inference on 20 validation images, three repeats, one excluded warm-up image, Intel Core i5-12500H (16 logical CPUs), Windows build 26200, PyTorch CPU 2.14.1, two torch threads, 96px model input. The cohort chunk size was eight, while the current engine performs individual model forwards at batch size **one**. Times include shared fundus preprocessing and both DR/quality model passes; they exclude JPEG file decoding, HTTP transport and application triage. Batch wall times additionally cover controller/output assembly.

| Strategy | Mean ms/image | Median ms/image | P95 ms/image | Extra-pass fraction |
| --- | ---: | ---: | ---: | ---: |
| Single pass | 230.94 | 227.17 | 328.39 | 0 |
| Second pass for all | 242.84 | 233.59 | 345.99 | 1.00 |
| Uncertainty trigger with cap | 289.73 | 262.79 | 458.95 | 0.25 |
| Severity trigger with cap | 280.75 | 248.36 | 490.63 | 0.25 |
| Ranked population budget | 231.05 | 217.85 | 342.37 | 0.25 |

The maximum observed RSS at batch boundaries was approximately 636–638 MiB. Timing varied under the live workstation workload; uncertainty/severity runs were slower despite the same activation cap. These small sequentially ordered measurements do not demonstrate a statistically reliable speed advantage or any clinical advantage. Image preprocessing dominates this low-resolution pilot workload. Holdout inference was reserved for the separate locked evaluation run.

## Reproduce

```powershell
.venv/Scripts/python.exe scripts/train_dr.py --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/dr --epochs 3 --image-size 96 --batch-size 8 --cpu-threads 2
.venv/Scripts/python.exe scripts/train_quality.py --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/quality --epochs 3 --image-size 96 --batch-size 8 --cpu-threads 2
.venv/Scripts/python.exe scripts/calibrate.py --dr-checkpoint checkpoints/dr/best.pt --quality-checkpoint checkpoints/quality/best.pt --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/calibration.json --cpu-threads 2
.venv/Scripts/python.exe scripts/benchmark_cpu.py --dr-checkpoint checkpoints/dr/best.pt --quality-checkpoint checkpoints/quality/best.pt --calibration checkpoints/calibration.json --manifest data/processed/deepdrid/pilot_manifest.csv --split validation --output artifacts/pilot_cpu_benchmark.json --cpu-threads 2 --batch-size 8 --repeats 3
```

Use held-out evaluation only after training/calibration and experimental choices are fixed. Do not choose thresholds or tune the model from final holdout results.

## Limits and next experiments

Larger permitted datasets, candidate-backbone/resolution comparison, patient-grouped uncertainty, natural quality-defect heads, external-domain evaluation, and finalized policy/compute selection still require experiments. A metadata `trained` marker establishes completed supervised software steps and does not establish accuracy. Calibration on 20 correlated-view images is highly uncertain. A CPU benchmark reports observed process RSS at batch boundaries rather than claiming a continuously sampled memory peak.

The implementation follows the official [torchvision ResNet-18 interface](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) for explicit weights selection and ImageNet normalization. Artifact loading uses the [PyTorch serialization API](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
