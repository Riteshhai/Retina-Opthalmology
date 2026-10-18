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

- Downloaded official DeepDRiD metadata, definitions and license pinned to
  56d8af71319803150d0e7b90533e0918c7bf85ee. Acquired260JPEGs (138.9MiB),
  65patients. Strict audit:260readable,0missing,0quarantinedpilotrows,
  no cross-split SHA or cropped texture pHash candidates. Full metadata has11
  ambiguous rows; these are quarantined. See docs/DATA_STATUS.md.
- Frozen manifest SHA256 e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0.
  Patient-disjoint160train/20selection/20calibration/60holdoutimages.
- Installed CPU torch2.14.1+cpu/torchvision0.29.1+cpu in local .venv.
- Executed scripts/train_dr.py and train_quality.py with epochs3,image-size96,
  batch-size8,seed42,2CPUthreads,randominitialization. Each completed60updates.
  Checkpoints selected on validation loss: DR epoch1,qualityepoch2.
- Executed scripts/calibrate.py on20dedicatedcalibrationimages, including TTA.
  Calibration artifacts are checksum-bound. DR NLL/Brier improved on fitting
  data while ECE worsened; quality T reached20. These are not independent metrics.
- Executed scripts/evaluate.py --manifest data/processed/deepdrid/pilot_manifest.csv
  --split holdout --output reports/pilot after freezing model/policy choices.
  Holdout60images15patients: DR accuracy16.67%,macroF1 .05714,kappa0,allgrade3;
  quality50%,allpoor. No useful screening performance or superiority established.
  Budget tables include51/63infeasibles and zero-coverage feasible joint settings.
  A joint .6review/.4recapture allocation sent7severe cases torecapture, versus0
  quality-only/naive cases, so no favorable policy conclusion is justified.
- Executed scripts/benchmark_cpu.py on validation20images ×3repeats. Source
  artifacts local; sanitized aggregate comparisons in reports/pilot. Machine
  was shared with other development activity, including container work; timings
  do not establish isolated performance or strategy superiority. RSS is sampled.
- scripts/run_ablations.py recomputed comparisons from frozen predictions without
  additional image/model inference;11reportingtests passed. Same conditional
  activation sets produced equal outputs; different timings are not better quality.
- Full pytest run with QSART_REAL_MANIFEST set:67passed in37.07s,including actual
  trained DR+quality+calibration,real JPEG upload,batch routing,TTAandCSV/JSONexport.
  One upstream AnyIO alias deprecation warning. New reporting subset and final
  aggregate run are recorded below when complete. Ruff passed.
- Actual HTTP localhost8001 readiness200 and real-image prediction200 with
  secondpass=true; all5Streamlitpages rendered,realAPIresponse table and export
  widgets verified with AppTest. Native browser displayed the running dashboard.
  Fixed stale single-image prediction association and temporary-upload wording.
- Docker Desktop29.7.2 started; initial CPU image built. Latest-source rebuild
  hit registry DNS and a wheel read timeout. Pinned the observed base digest,
  split/cache dependency layers and extended pip retries/timeouts. Verification
  of final rebuilt containers remains pending at this log entry.
- Literature audit:43records,all28originalIDs retained;43BibTeXentries with
  source-specific verification limits. Review1brief,manuscriptdraft and Mermaid
  architecture reflect actual negative pilot findings. Originals preserved.
- Training checkpoint code_version captured the pre-existing HEAD while source
  was uncommitted. It is not the exact training source identifier; this limitation
  is explicit in docs/MODEL_STATUS.md. Heldout report additionally hashes source.
- Raw images,patient manifests,checkpoints,per-image artifacts and Office source
  extracts remain local and excluded from Git. Source and aggregate reports are
  prepared for a reviewable GitHub branch; main is not modified.
