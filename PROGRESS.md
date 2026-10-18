# Execution log

## 2026-10-18 — audit and bootstrap

- Inspected parent directory and all starter ZIP member names; read starter
  README, data parser, policy, CLI, tests and research protocol.
- Queried GitHub repository inventory and target contents through the connected
  GitHub plugin; the user selected Riteshhai/Retina-Opthalmology.
- Preserved originals; extracted starter to QSART; preserved full build prompt.
- Python 3.11 detected; torch/torchvision absent initially. No project retinal
  images or trained checkpoint found at initial audit.
- Parallel work: dataset ingestion; model/training/calibration; source and
  literature verification. Root integrates policy, API, dashboard and packaging.
- Real-image training and deployment are not yet verified.

## 2026-10-18 — genuine pilot and application verification

- Downloaded official DeepDRiD metadata, annotation definitions and license at
  commit 56d8af71319803150d0e7b90533e0918c7bf85ee. Acquired 260 JPEGs
  (138.9 MiB), representing 65 patients. Strict audit: all readable, no missing
  or quarantined pilot images, no cross-split SHA or texture pHash candidates.
  Eleven ambiguous rows in the full annotations are quarantined.
- Frozen manifest SHA256:
  e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0.
  Patient-disjoint partitions: 160 training, 20 checkpoint-selection validation,
  20 calibration and 60 holdout images. See docs/DATA_STATUS.md.
- Installed torch 2.14.1+cpu and torchvision 0.29.1+cpu in the local .venv.
- Executed scripts/train_dr.py and train_quality.py with 3 epochs, 96-pixel input,
  batch size 8, seed 42 and 2 CPU threads, from random initialization. Each
  completed 60 optimizer updates. Best validation-loss checkpoints: DR epoch 1,
  quality epoch 2. See docs/MODEL_STATUS.md for exact commands and fitting limits.
- Executed scripts/calibrate.py on 20 dedicated calibration images, including
  separate flip-TTA temperatures. Calibration artifacts are checksum-bound.
  DR fitting NLL/Brier improved while ECE worsened; quality temperature reached
  20. These fitting metrics are not independent test results.
- Executed scripts/evaluate.py --manifest data/processed/deepdrid/pilot_manifest.csv
  --split holdout --output reports/pilot after freezing model and policy choices.
  Holdout: 60 images / 15 patients. DR accuracy 16.67%, macro-F1 0.05714, kappa 0;
  every prediction was grade 3. Quality accuracy 50%; every prediction was poor.
  Useful screening performance and policy superiority were not established.
- Seven policy comparisons across nine budget scenarios produced 51 infeasible
  cases. A joint allocation at review 0.6 / recapture 0.4 gave zero auto coverage
  and sent 7 severe cases to recapture, versus 0 for quality-only/naive policies.
  These unfavorable findings are preserved. No holdout-based retuning occurred.
- Executed scripts/benchmark_cpu.py on 20 validation images, three repeats.
  Source artifacts remain local; sanitized comparisons are in reports/pilot.
  Other development and container activity shared the machine during measurement;
  these timings do not establish isolated performance or strategy superiority.
  Inference forwards one image at a time; RSS is sampled at batch boundaries.
- scripts/run_ablations.py recomputed aggregate comparisons from frozen predictions
  without extra model inference. Its 11 reporting tests passed. The three
  conditional strategies selected identical cases and produced equal outputs.
- Full pytest run with QSART_REAL_MANIFEST set: 67 passed in 37.07 seconds,
  including a genuine trained-model JPEG upload, quality/calibration, TTA,
  batch routing and JSON/CSV export. One upstream AnyIO deprecation warning.
  Later added reporting tests and the final aggregate run are recorded below.
  Ruff checks passed.
- Actual HTTP readiness and prediction on localhost:8001 returned 200 using a
  real DeepDRiD image and second_pass=true. All five Streamlit pages rendered;
  the real response table and export widgets were verified with AppTest.
  The browser displayed the running dashboard. Fixed stale prediction/image
  association and accurately described temporary upload handling.
- Docker Desktop 29.7.2 started; an initial CPU image built. Latest-source
  rebuilds encountered registry DNS and a wheel read timeout. Pinned the observed
  base digest, cached separate dependency layers, and extended pip retries and
  timeouts. Final container execution is pending at this log entry.
- Literature: 43 records and 43 BibTeX entries, retaining all 28 original IDs,
  with source-specific verification limits. Updated the Review-1 brief,
  manuscript draft and Mermaid architecture with actual negative pilot findings.
- Checkpoint code_version records pre-existing HEAD while new source was
  uncommitted. It is not an exact training source identifier; this is documented.
  The held-out report additionally hashes source. Initial implementation commit:
  330bb87 (recorded after pilot execution).
- Raw images, patient manifests, weights, per-image artifacts and Office source
  extracts stay local and are ignored by Git. Source and aggregate reports are
  committed on qsart/research-prototype; main remains unchanged.

## 2026-10-18 — final verification and delivery

- Final aggregate suite with real-image opt-in: 78 passed, no failures/skips,
  127.03 seconds. One upstream AnyIO deprecation warning. Ruff passed.
  Summary: reports/test_summary.json; full JUnit stays local.
- Latest CPU Docker image built after network retry fixes. Both containers are
  healthy. Original host port 8000 belongs to another Python application, so
  Compose ports are configurable and the ignored local .env selects 18000/18501.
  No unrelated application was stopped.
- scripts/verify_deployment.py exercised the actual containers: health, readiness,
  model info, docs, real JPEG prediction, 2-image batch, conditional second pass,
  upload rejection and dashboard health all passed. After restarting the API,
  the same verification passed again, confirming checkpoint persistence.
- Container and host hashes match all 27 runtime Python files. Captured the
  actual container package versions in reports/container_requirements.txt.
- Browser displayed the running container dashboard; saved reports/dashboard_preview.png.
  Removed only the temporary development API/dashboard processes on 8001/8502.
  The verified Docker services remain running on 18000/18501.
- PowerShell launcher syntax parsed successfully. It reads local port settings
  and cleans up the child dashboard when its foreground API exits. The launcher
  itself was not used for final Docker verification.
- Source and aggregate evidence published to qsart/research-prototype; draft PR:
  https://github.com/Riteshhai/Retina-Opthalmology/pull/1 . Main remains unchanged.
- Acceptance remains PARTIALLY COMPLETE for the overall research project.
  The local engineering pilot is verified; useful model performance, complete
  confirmatory experiments, natural-defect modeling and external validation
  remain open. No clinical or scientific superiority is claimed.
