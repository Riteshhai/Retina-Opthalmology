# QSART — Quality- and Severity-Aware Retinal Triage

A local, CPU-compatible **research prototype** with genuine pilot-trained DR and image-quality models, temperature calibration, three-action triage, independent cohort budgets, a FastAPI service and a Streamlit client.

**Status: PARTIALLY COMPLETE.** The initial real-image engineering pipeline works; it does not establish useful screening performance or clinical validity. The first 96-pixel, three-epoch, randomly initialized DR model predicted one grade on the 60-image pilot holdout (accuracy 16.67%, macro-F1 0.0571, kappa 0). It must not be used for patient care. Full research experiments and external validation remain outstanding. See [acceptance evidence](docs/ACCEPTANCE_STATUS.md), [pilot report](reports/pilot/evaluation.json), and [model limitations](docs/MODEL_STATUS.md).

All local work lives in this QSART directory inside `MAJOR PROJECT CV`. Original Office documents, bibliography files and starter ZIP remain unchanged in its parent directory. Datasets, patient manifests, trained weights and raw per-image artifacts are intentionally local and ignored by Git.

## Start the application

With Docker Desktop running and local model artifacts available:

```powershell
docker compose up --build -d
```

- Dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs
- Liveness: http://localhost:8000/health
- Trained-artifact readiness: http://localhost:8000/ready
- Stop: `docker compose down`

The API is alive without checkpoints, but readiness and prediction fail explicitly. There is no generated or random prediction fallback. Compose mounts `./checkpoints` read-only; it never downloads weights at startup. Ports are bound to the local machine. The API and dashboard have independent health checks; liveness does not imply model readiness.

Required artifacts: `checkpoints/dr/best.pt`, `checkpoints/quality/best.pt`, `checkpoints/calibration.json`. The calibration artifact is bound to checkpoint and manifest checksums. Test fixtures are refused by production loading.

## Local Python setup

Python 3.11 is the verified version. CPU-only wheels require the PyTorch index:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
# In a second terminal:
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Alternatively, `./scripts/start_local.ps1` launches both local services. Use either Compose or local processes to avoid port conflicts. `.env.example` lists settings; export them for local Python. The API accepts `QSART_DR_CHECKPOINT`, `QSART_QUALITY_CHECKPOINT`, `QSART_CALIBRATION`, and `QSART_CPU_THREADS`; the dashboard uses `QSART_API_URL` and `QSART_REPORTS_DIR`. Compose supplies container paths explicitly.

Direct dependencies are pinned in requirements.txt; Docker resolves compatible transitive dependencies and its executed environment is captured in the deployment report. The development venv initially inherited existing host libraries, while the container provides an independent clean installation. No paid service is required.

## Reproduce the real-image pilot

The official DeepDRiD repository is pinned to commit `56d8af71319803150d0e7b90533e0918c7bf85ee`. Read [data terms](docs/research/DATA_ACCESS_AND_TERMS.md) and [schema](docs/DATA_SCHEMA.md). Images are downloaded locally from the author repository, with attribution and per-file checksums; no private archive or gated access is bypassed.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --download-pilot --out data/processed/deepdrid/pilot_manifest.csv
.\.venv\Scripts\python.exe scripts/validate_data.py data/processed/deepdrid/pilot_manifest.csv --out data/processed/deepdrid/pilot_manifest.verified_audit.json
.\.venv\Scripts\python.exe scripts/train_dr.py --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/dr --epochs 3 --image-size 96 --batch-size 8
.\.venv\Scripts\python.exe scripts/train_quality.py --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/quality --epochs 3 --image-size 96 --batch-size 8
.\.venv\Scripts\python.exe scripts/calibrate.py --dr-checkpoint checkpoints/dr/best.pt --quality-checkpoint checkpoints/quality/best.pt --manifest data/processed/deepdrid/pilot_manifest.csv --output checkpoints/calibration.json
.\.venv\Scripts\python.exe scripts/evaluate.py --manifest data/processed/deepdrid/pilot_manifest.csv --split holdout --output reports/pilot
```

Pilot: 260 images / 65 patients, 138.9 MiB downloaded; train 160 / 40 patients, checkpoint-selection validation 20 / 5, calibration 20 / 5, held-out pilot 60 / 15. Sampling was predefined by maximum patient grade, so it cannot estimate screening prevalence. Official validation supplies the pilot holdout; official training supplies the other three disjoint partitions. Eleven ambiguous source annotation rows are quarantined rather than silently corrected. The pilot excludes their subjects.

The initial manifest is frozen and hash-recorded. Shared cropping and normalization are used during training and inference. Quality 0 means not good enough, 1 means good enough. Detailed ordinal defect annotations are preserved; the initial trained quality model uses the binary overall label. They are not equivalent to the scalar model probability.

The training CLI supports optional explicitly requested pretrained weights, resume, class balancing, early stopping, validation selection and CPU/CUDA configuration. This run used no pretrained download. Training a production-quality model needs a substantially larger, independently evaluated experiment. Keep all consumed pilot evaluation data separate from a future confirmatory test.

## Research policy and evaluation

`AUTO_SCREEN` returns a provisional five-grade result, including positive grades. It does not mean disease-free. `RECAPTURE` and `SPECIALIST_REVIEW` are unresolved recommendations, not completed care.

Batch routing solves an assignment problem with separate `floor(N * budget)` review and recapture capacities. Low-quality images cannot receive AUTO_SCREEN. If their required deferrals exceed capacity, the service reports infeasibility (HTTP 422); it never silently relaxes budgets. Single-image routing uses fixed thresholds and explicitly has no population workload guarantee. [Policy mathematics and exploratory cost assumptions](docs/TRIAGE_METHOD.md) describe these choices.

Seven baseline/proposed policies, budget grids, quality-by-severity error analysis, calibration comparisons and ablations produce actual aggregate pilot artifacts. Same budget ceilings do not guarantee matched realized workloads or demonstrate superiority. Primary metrics distinguish automated errors and severe cases deferred to recapture. Empty metric denominators are null. Patient-group bootstrap intervals acknowledge correlated views.

Conditional inference uses horizontal-flip logit averaging and dedicated TTA calibration; the per-request second-pass cap is rounded down. The validation benchmark compares single pass, all-image second pass, uncertainty triggers, severity triggers and budgeted selection. It measures CPU timings rather than claiming a speed advantage. Re-run with:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_cpu.py --dr-checkpoint checkpoints/dr/best.pt --quality-checkpoint checkpoints/quality/best.pt --calibration checkpoints/calibration.json --manifest data/processed/deepdrid/pilot_manifest.csv --split validation --output artifacts/pilot_cpu_benchmark.json --batch-size 8 --repeats 3
```

The inference loop forwards one image at a time; benchmark batch size means request cohort chunk size. RSS is sampled, not a continuous peak. See the report for CPU, thread count, resolution and timing variability. Larger tests, mapped natural-defect heads, robustness experiments, matched-workload statistical comparisons, model selection across backbones, and independent external-domain testing remain research work.

## API and interface

Endpoints: `GET /health`, `/ready`, `/model/info`, `/docs`; `POST /predict` (multipart `file`); `POST /predict/batch` (multipart `files`, query `review_budget`, `recapture_budget`, `second_pass_budget`). PNG/JPEG only; maximum 12 MiB per image, 24 million pixels, minimum side 32 pixels, maximum batch 64. Correct shape/format does not establish that a photograph is a retina or in distribution.

The dashboard displays actual current-session results, image preprocessing, probabilities, quality, entropy, actions, CPU timings, model information and saved experiment artifacts. It supports multi-image upload, action/grade filters and CSV/JSON export. Uploads are temporary and closed after processing; the application does not retain source images. Large multipart uploads may use temporary spooled files. Access logs are disabled in provided server commands, and application errors omit image contents and original filenames.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
$env:QSART_REAL_MANIFEST='data/processed/deepdrid/pilot_manifest.csv'
.\.venv\Scripts\python.exe -m pytest tests/test_real_integration.py -q
docker compose config --quiet
```

Ordinary tests use explicitly labeled software fixtures and do not require the Internet. Real-image integration is opt-in and never substitutes synthetic images. See PROGRESS.md and reports for actual executed results. Tests do not imply clinical validation.

## Repository map and preserved assets

- `qsart/data`, `preprocessing`: acquisition adapters, manifests, integrity and preprocessing.
- `qsart/models`, `training`, `calibration`, `inference`: genuine PyTorch pipeline and artifact validation.
- `qsart/triage.py`, `evaluation.py`: budget allocator and held-out reporting.
- `api`, `dashboard`: service and client.
- `scripts`, `configs`, `tests`, `.github/workflows`: reproducible commands and verification.
- `reports/pilot`: aggregate measured results and figures.
- `docs/research`: 43-record literature audit, bibliography, protocol, Review-1 briefing and manuscript draft.
- `PROJECT_AUDIT.md`, `TASKS.md`, `PROGRESS.md`: status and evidence.

The original starter `app.py`, `qsart/policy.py`, `qsart/cli.py` remain a separate **synthetic-score policy bench**. Its `routine` means auto-clearance, unlike the new provisional AUTO_SCREEN action. Its legacy tuning constrains calibration workloads, not future cohorts. Do not use its built-in mock CSVs as clinical results. See docs/LEGACY_STARTER_README.md.

No broad novelty or clinical benefit is claimed. Close prior work already covers quality-aware grading, ordinal uncertainty and screening integration. The manuscript is a research draft; its confirmatory results remain incomplete. GitHub is used for source history and review, while local images, manifests, training logs and checkpoints remain outside version control.
