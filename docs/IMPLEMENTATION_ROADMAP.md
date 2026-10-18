# Project milestones — academic outcomes and deliverables

Proposed work packages; actual dates must be aligned with semester/course submission and venue deadlines. CPU feasibility depends on RAM, storage, model and dataset sizes; a mid-level laptop may be sufficient for small-model fine-tuning and inference but full-size fundus-model training can be slow.

## Phase 0 — now / completed in starter
- Verified core competing methods & dataset sources, correct novelty positioning.
- Offline policy simulator, calibration-only threshold search, metrics, unit tests, synthetic benchmark UI, Docker starter.
- No retinal image model trained, no real empirical evidence produced.

## Phase 1 — dataset audit + reproducible baselines
- Download official DeepDRiD and read labels, license, patient identifiers, studies.
- Keep images/eyes from the same patient in same split; inspect class distributions by quality and severity.
- Document SHA256 manifest, artifacts excluded, uncertain labels, cohort/source overlap.
- Fine-tune compact baseline MobileNetV3 / EfficientNet-B0 from licensed pretrained weights (weights downloading needs internet); multiclass DR and binary referable-head experiment.
- Add a quality model; export per-image probabilities on disjoint splits.
- Success gate: reliable DR and quality baseline with explicit held-out metrics and reproducibility.

## Phase 2 — primary hypothesis
- Extend the existing *independently tuned* risk-only, quality-first, confidence-only, and severity-aware policies with rate-matched Pareto-frontier comparisons.
- Budget feasibility grid; frozen calibration on internal data; evaluate internal test and external cohorts.
- Cluster bootstrap and paired significance; risk–coverage and referral–recapture Pareto frontier; secondary penalties for severe disease sent to recapture instead of review.
- Success gate: meaningful reproducible reduction in severe missed auto-clear cases at matched feasible constraints, OR properly documented null finding.

## Phase 3 — supporting contributions
- Quality components (artifacts, clarity and field-of-view) and per-stratum ablation.
- Optional conditional second pass for ambiguous images; CPU latency on named CPU, peak RAM and utilization; ONNX only if conversion works.
- Reliability diagrams and calibration/abstention shifts on external images; note unavailable quality labels in IDRiD.

## Phase 4 — working application and release
- Image preprocessing pipeline (reversible; preserve originals), offline local API, Streamlit demo, Docker Compose with fixed package versions.
- UI: image upload, separate quality and DR outputs, safe policy status, threshold/uncertainty explanation, unknown/out-of-domain warnings, model and dataset version, audit event log.
- Batch inference/export, container health endpoint, failure/large-file handling, unit/integration tests and offline CPU benchmarking.
- Clearly label 'research use only / not for clinical decisions'. Clinical deployment requires external validation and regulatory review.

## Phase 5 — paper submission
- Use CLAIM 2024 reporting checklist; draft methods while experiments are in progress, results only after data are locked.
- Present comparative safety/coverage curves, stratified failures, ablations, external generalization and compute costs with CIs.
- Seek domain expert advice and institutional ethics statement; check dataset and trained-weight redistribution policies.
- Check conference/journal fit and real publication/registration costs before submission.

## Possible venues with verified dates (as of 2026-10-18)
- ISBI 2027 full-paper deadline **2026-10-26**: only eight days away; NOT realistic for an unimplemented novel empirical paper. In-person accepted presentation required in Lausanne, so travel/registration costs would apply. https://biomedicalimaging.org/2027/papers/
- EMBC 2027 full contributed papers deadline **2027-01-24**, notification **2027-04-16**, final version **2027-04-30**. Four technical pages + one references page. Ambitious *submission target* if experiments and manuscript are complete; not a guarantee of acceptance. https://embc.embs.org/2027/papers/
- For no/low author-side publication fees, investigate traditional subscription-based journal tracks and page limits; do not assume all IEEE journals/conferences are free (https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/about-potential-article-processing-charges/).

## Four review gates
1. VIT supervisor approves scope and endpoint before expensive training.
2. Dataset provenance/splits auditable before any model-performance claims.
3. Baselines independently tuned and statistical analysis frozen before external evaluation.
4. Submission only when artifacts, evidence, manuscript, ethics, and venue requirements are verified.
