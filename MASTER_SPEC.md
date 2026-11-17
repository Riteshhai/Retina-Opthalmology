# QSART — MASTER PROJECT BUILD SPECIFICATION

**Version:** 1.0 — Living Master Prompt  
**Created:** October 20, 2026  
**Intensive Development Window:** October 21 – November 3, 2026  
**Project Type:** Full-Stack AI/ML Research and Deployment System  
**Academic Program:** M.Tech Artificial Intelligence & Machine Learning, VIT Vellore  
**Project Requirement:** 10-credit outcome-based research project  
**Primary Execution Agents:** OpenAI Codex + GPT Work  
**Research Paper:** Formal writing after implementation and experimental evaluation  
**Development Philosophy:** Build fast. Build reliably. Make it impressive. Measure everything.

---

# PART I — PROJECT VISION AND INSTRUCTIONS

## 1. YOUR ROLE

You are operating as an elite multidisciplinary AI research and engineering team responsible for building a complete, research-grade medical computer vision platform.

Simultaneously act as:

1. Principal AI Research Scientist
2. Medical Computer Vision Researcher
3. Senior Machine Learning Engineer
4. MLOps and Deployment Engineer
5. Backend Architect
6. Frontend and UX Engineer
7. Data Engineer
8. Software Quality Assurance Engineer
9. Experimental Research Analyst
10. Technical Documentation Engineer

Your mission is not to generate disconnected source files or merely suggest implementation steps.

**Your mission is to deliver an integrated, working, experimentally validated, locally deployable software product with a defensible research contribution.**

Act as a senior engineering team with high technical standards, rapid execution, and ownership of the complete outcome.

Use the user's existing QSART project materials, prior research decisions, literature review, presentation materials, starter implementations, and available repositories whenever accessible.

Do not unnecessarily start from scratch.

## 2. PROJECT NAME

**QSART — Quality- and Severity-Aware Retinal Triage**

Extended research title:

**Quality- and Severity-Aware Risk-Constrained Retinal Triage for Reliable Diabetic Retinopathy Screening Under Image Degradation and Domain Shift**

Working publication title:

**QSART: Dual-Budget Severity-Aware Selective Screening of Diabetic Retinopathy with Quality-Guided Triage and Resource-Constrained Inference**

The publication title may evolve as results become available.

## 3. PRIMARY OBJECTIVE

Develop an AI-assisted retinal screening research system capable of:

- Analyzing actual retinal fundus photographs.
- Predicting diabetic retinopathy severity.
- Assessing retinal image quality.
- Quantifying prediction uncertainty.
- Using ordinal conformal prediction where appropriate.
- Prioritizing severe and referable disease.
- Choosing appropriate automated screening, recapture, or specialist-review actions.
- Respecting independently defined resource budgets where feasible.
- Performing additional inference only when beneficial.
- Operating efficiently on ordinary CPU hardware.
- Providing a professional clinician-oriented screening workflow.
- Generating reproducible research experiments.
- Running through Docker with a functioning API and frontend.

The central scientific question is:

**Can quality-aware, severity-sensitive, uncertainty-guided decision policies reduce missed referable diabetic retinopathy under independent review and recapture constraints, while maintaining practical CPU inference performance?**

Treat this as a hypothesis requiring experimental validation, not an established result.

## 4. ABSOLUTE REQUIREMENTS

These requirements are mandatory:

**R1 — Working implementation:** Every essential feature must execute successfully, not merely exist in source code.

**R2 — Real trained AI:** Final predictions must come from genuinely trained or legitimately fine-tuned models using real permitted retinal images.

**R3 — Deployability:** A complete CPU-compatible Docker deployment with FastAPI and Streamlit is mandatory.

**R4 — Zero additional cost:** Avoid paid APIs, commercial infrastructure, subscription dependencies, and mandatory cloud resources.

**R5 — Mid-range laptop compatibility:** The deployed inference application must run on typical consumer CPU hardware. GPU use is optional for training.

**R6 — Research integrity:** All benchmark results, metrics, graphs, and comparisons must originate from actual experiments.

**R7 — Novelty implementation:** The system must implement and evaluate all three selected research contributions.

**R8 — Reproducibility:** Experiments must preserve data manifests, seeds, configurations, code revisions, checkpoint information, and actual outputs.

**R9 — Demonstrability:** The final product must support a complete real-image demonstration from upload through prediction and triage.

**R10 — Paper readiness:** The research methodology and experimental evidence must be sufficient to support subsequent formal manuscript preparation.

**R11 — Medical responsibility:** This is a research prototype, not a clinically approved diagnostic system. Do not make claims of clinical readiness or validated safety without appropriate evidence.

**R12 — No silent substitutes:** If required model files, datasets, permissions, or dependencies are missing, report the missing components rather than silently replacing them with fabricated functionality.

---

# PART II — SCIENTIFIC FOUNDATION

## 5. BASE RESEARCH PAPER

The primary foundation is:

**Hasan, U., Nayeem, M. A., and Alghamdi, T. G. (2026).**

*Uncertainty-Aware AI-Assisted Diabetic Retinopathy Grading from Fundus Images with Ordinal Conformal Prediction.*

Diagnostics, 16(16), 2645.

DOI:

https://doi.org/10.3390/diagnostics16162645

### Required base-paper study

Read and critically analyze the entire paper.

Extract:

- Research question.
- Dataset choices.
- Preprocessing protocol.
- Neural network architectures.
- Training and validation methodology.
- Calibration procedure.
- Standard conformal prediction methods.
- Ordinal prediction set construction.
- Internal and external validation.
- Coverage and set-size metrics.
- Failure modes under dataset shift.
- Limitations and future research directions.

The base paper evaluates uncertainty-aware grading using architectures including EfficientNet-B0, ResNet-50, and ViT, with APTOS as source data and external evaluations involving IDRiD and Messidor-2.

Its main relevance to QSART is uncertainty reliability and ordinal severity representation under domain shift.

### Our extension

We are not simply reproducing the base paper.

We extend uncertainty-aware DR grading into an operational triage framework that considers:

1. Separate review and recapture capacity constraints.
2. Severity-weighted consequences of screening errors.
3. Image-quality effects on decision reliability.
4. Selective computational effort.
5. End-to-end deployment feasibility.

Distinguish clearly between a reproduced baseline, an inspired implementation, and a genuinely novel method.

If exact reproduction is impossible because of missing code, datasets, hyperparameters, or computational resources, document the differences.

Never falsely describe an approximate replication as an exact reproduction.

## 6. FINALIZED RESEARCH NOVELTIES

### N1 — Dual-Budget Severity-Weighted Selective Triage

**Priority: Critical. This is the primary novelty.**

Develop a three-action decision framework.

Action 1:

`AUTO_NONREFERABLE`

The system considers the image sufficiently reliable for a provisional automated non-referable screening outcome.

Action 2:

`RECAPTURE`

The system recommends new image acquisition because image quality or image-related diagnostic reliability is inadequate.

Action 3:

`SPECIALIST_REVIEW`

The system defers interpretation to a qualified specialist due to possible referable disease, severe risk, uncertainty, or another safety-relevant concern.

Two independent budgets must be modeled:

- `B_review`: Maximum specialist-review workload.
- `B_recapture`: Maximum recapture workload.

The research objective is to optimize screening risk subject to these resource constraints.

Use configurable severity-weighted error costs.

Give particular attention to missed moderate-or-worse DR, severe DR, and proliferative DR.

Do not assume all mistakes are clinically equivalent.

#### Mathematical formulation

For N evaluated images and an action policy π:

Minimize:

`Expected severity-weighted screening risk`

Subject to:

`N_review / N <= B_review`

`N_recapture / N <= B_recapture`

And:

`action_i ∈ {AUTO_NONREFERABLE, RECAPTURE, SPECIALIST_REVIEW}`

Additional risk constraints may be investigated.

Implement an empirical optimization procedure using the validation partition.

Compare against:

- No-deferral baseline.
- Confidence-only referral.
- Entropy-only referral.
- Severity-only referral.
- Image-quality threshold policy.
- Single-budget review policy.
- Simple quality-plus-uncertainty policy.
- Proposed dual-budget severity-aware policy.

Use matched operating constraints.

Develop decision thresholds and risk scores based only on training, calibration, and validation data.

For independent test sets, apply frozen policies.

If a safe operational allocation is impossible under the configured budgets, explicitly report an infeasible or overloaded state.

Never silently classify a high-risk or ungradable case as non-referable simply to meet a budget.

Clearly differentiate recommended actions from actual completed clinician decisions or successful recaptures.

### N2 — Quality × Severity Robustness and Failure Analysis

**Priority: High. Supporting research novelty.**

Study relationships between:

- Image quality.
- Disease severity.
- Prediction uncertainty.
- Model correctness.
- Sensitivity to referable DR.
- Screening action.

Build an evidence-based reliability analysis across severity classes and quality conditions.

Potential image-quality dimensions include:

- Overall quality.
- Blur.
- Illumination.
- Artifacts.
- Field definition.
- Other verified annotation dimensions.

Do not assume all datasets contain these labels.

Use DeepDRiD's actual available quality annotations and optionally EyeQ for additional supervision.

Test genuine low-quality images and controlled synthetic degradation separately.

Generate:

- Quality-versus-severity error heatmaps.
- Per-class confusion matrices.
- Risk-versus-quality plots.
- Uncertainty-versus-degradation curves.
- Missed-referable-case analyses.
- Stratified calibration results.

Quantify whether the proposed policy offers improvements across these conditions.

### N3 — CPU-Efficient Conditional Inference

**Priority: High. Supporting engineering/research novelty.**

Develop a two-stage inference framework.

First pass:

Run a lightweight retinal classifier and estimate severity, image quality, and uncertainty.

Second pass:

For selected uncertain, borderline, or high-risk cases, perform additional inference.

Candidate techniques:

- Higher-resolution evaluation.
- A second lightweight classifier.
- Test-time augmentation.
- Alternative retinal crop.
- Lightweight ensemble.

Start with a straightforward technique and measure whether it adds value.

Use a configurable CPU-inference budget.

Evaluate:

- Base model alone.
- Enhanced inference for every image.
- Confidence-triggered enhanced inference.
- Risk-triggered enhanced inference.
- Proposed budget-aware conditional inference.

Record the additional inference fraction, mean latency, P50/P95 latency, RAM, and changes in screening metrics.

Do not assume the second pass improves predictions.

Retain it as a claimed contribution only if the experiments justify its usefulness.

---

# PART III — DATASETS AND DATA ENGINEERING

## 7. DATASET ROADMAP

### Dataset A — APTOS 2019

Role: Primary grading-model development and base-paper-aligned experiments.

Source:

https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Use the available labeled training images and five-level DR severity annotations.

Do not assume patient identifiers or quality labels are available.

Do not treat Kaggle's competition test labels as accessible when they are not.

Plan a defensible internal development/calibration/test protocol.

### Dataset B — DeepDRiD

Role: Joint severity and quality experiments, including quality-failure analysis.

Source:

https://github.com/deepdrdoc/DeepDRiD

Use verified DR and quality labels.

Inspect subject, eye, view, and quality metadata.

Avoid leakage between views of the same eye or patient.

Preserve the distinction between image-level, eye-level, and patient-level annotations.

### Dataset C — EyeQ

Role: Supplemental image-quality modeling.

Source:

https://github.com/HzFu/EyeQ

Quality categories:

- Good
- Usable
- Reject

Check access to the original image files separately from the annotation repository.

Do not assume pretrained quality checkpoints are usable without verification.

### Dataset D — IDRiD

Role: Held-out external evaluation and possible supplemental research analysis.

Source:

https://idrid.grand-challenge.org/

Preserve IDRiD as an independent evaluation dataset for the primary external-domain experiment.

Do not tune model parameters or final policy thresholds on this held-out evaluation set.

Keep DR grading and DME labeling separate.

### Optional datasets

- DDR: https://github.com/nkicsl/DDR-dataset
- Messidor-2: Use only after confirming image and compatible label access.

Additional datasets must support a specific experimental objective and must not delay the initial working prototype.

## 8. DATA PIPELINE

Implement real modules for:

1. Dataset discovery.
2. Source validation.
3. Label parsing.
4. Image integrity verification.
5. Duplicate detection.
6. Group-aware train/validation/calibration/test splitting.
7. Fundus border detection and cropping.
8. Resize and color preprocessing.
9. Class-balanced sampling or weighting.
10. Dataset visualization.
11. Data provenance.
12. Manifest generation.

Maintain raw source labels separately from standardized labels.

Record any cross-dataset mapping.

Do not merge incompatible quality labels without a justified transformation.

### Data leakage prevention

Prefer subject-disjoint splits when authoritative identifiers exist.

Where identifiers are unavailable, document limitations and conduct filename, exact-hash, and perceptual-near-duplicate checks.

Never tune final policy thresholds against the evaluation dataset.

### Dataset outputs

Generate:

- `dataset_manifest.csv`
- `dataset_summary.json`
- `class_distribution.png`
- `quality_distribution.png`
- `split_report.json`
- `duplicates_report.csv`
- `data_validation_report.md`

Do not invent entries or silently fabricate missing metadata.

---

# PART IV — AI MODEL SYSTEM

## 9. CORE ARCHITECTURE

Implement the complete system around the following logical pipeline:

```text
Retinal Fundus Image
         |
         v
Image Validation
         |
         v
Fundus Preprocessing
         |
         v
+---------------------------+
| Lightweight DR Predictor  |
| EfficientNet-B0 / similar |
+---------------------------+
         |
         +------------------------+
         |                        |
         v                        v
DR Severity Probabilities    Quality Predictor
         |                        |
         v                        |
Calibration + Ordinal CP <-------+
         |
         v
Uncertainty and Risk Estimation
         |
         v
Conditional Second-Pass Controller
         |
         v
Updated Risk Assessment
         |
         v
Dual-Budget Severity-Aware Policy
         |
         v
+--------------------------------+
| AUTO_NONREFERABLE              |
| RECAPTURE                      |
| SPECIALIST_REVIEW              |
+--------------------------------+
         |
         v
FastAPI Inference Service
         |
         v
Streamlit Research Dashboard
         |
         v
Screening Report and Analytics
```

The implementation may improve the precise execution order, but these logical modules must remain independently testable.

## 10. DR CLASSIFICATION MODEL

Default architecture:

**EfficientNet-B0**

Alternative lightweight comparison:

**MobileNetV3 or ResNet-18**

Predict five DR severity grades:

- Grade 0: No DR.
- Grade 1: Mild DR.
- Grade 2: Moderate DR.
- Grade 3: Severe DR.
- Grade 4: Proliferative DR.

Use a configurable referable threshold, initially grade 2 or above, and document its operational interpretation.

Train using genuine retinal fundus images.

Implement:

- Transfer learning.
- Optional staged backbone unfreezing.
- Class-weighted loss or another justified imbalance treatment.
- Early stopping.
- Learning-rate scheduling.
- Model checkpointing.
- Reproducible seeds.
- Checkpoint metadata.
- Validation-driven model selection.
- CPU-compatible model export.

Do not require a massive transformer model for the first deployable version.

## 11. IMAGE QUALITY MODULE

Implement quality assessment using a verified labeled dataset.

Prefer an initial small model or compact quality head.

Support binary or ordinal quality classification according to the training source.

For DeepDRiD, support the real quality dimensions available in the annotations.

For EyeQ, preserve Good/Usable/Reject semantics.

Do not pretend predictions of detailed defect categories are supervised if only overall image quality is labeled.

The quality module must produce a calibrated or interpretable reliability score suitable for the triage policy.

## 12. UNCERTAINTY AND CONFORMAL MODULE

Implement:

- Softmax confidence.
- Predictive entropy.
- Temperature scaling.
- Brier score.
- Expected calibration error.
- Reliability diagrams.
- At least one conventional conformal prediction baseline.
- Ordinal conformal prediction aligned with the base paper where feasible.

Use separate calibration data.

For ordinal conformal prediction, preserve the severity ordering and implement the documented set-construction procedure.

Report:

- Marginal coverage.
- Severity-conditional empirical coverage.
- Prediction-set size.
- Set-size distributions.
- Severe-class undercoverage.
- Behavior under domain shift.

Do not claim formal conformal coverage guarantees on shifted external data without the assumptions required to justify them.

Compare uncertainty methods using actual evaluated outputs.

## 13. COMPLETE TRAINING PIPELINE

Support:

- CPU training for small experiments.
- Optional CUDA acceleration.
- Configurable image size.
- Configurable batch size.
- Training checkpoints.
- Resume training.
- Training/validation metric histories.
- Class imbalance monitoring.
- Numerical stability checks.
- Model versioning.

Training should be executable through documented commands.

The default inference application must not train models at startup.

Never replace absent trained weights with random weights while presenting the predictions as real.

---

# PART V — COMPLETE SOFTWARE PRODUCT

## 14. APPLICATION VISION

Build QSART as a professional AI-assisted retinal screening research platform.

The UI should appear polished, consistent, and credible for an academic medical-AI demonstration.

Suggested design language:

- Clean clinical interface.
- Dark/light mode support where practical.
- Blue, teal, and neutral colors.
- Readable medical images.
- Crisp visual hierarchy.
- Well-organized analytical charts.
- Responsive layouts.
- Clear risk and uncertainty indicators.
- Minimal distracting animation.

Avoid an interface that looks like a collection of disconnected Streamlit examples.

The interface must be backed by actual functioning APIs and research artifacts.

## 15. REQUIRED DASHBOARD MODULES

### Module A — Overview Dashboard

Show:

- Total processed images.
- Provisional DR grade distribution.
- Image quality distribution.
- Automated non-referable count.
- Recapture recommendation count.
- Specialist-review recommendation count.
- Screening workload statistics.
- Actual model version.
- Processing latency summaries.

Every displayed value must be computed from actual available records.

### Module B — Single Retinal Image Analysis

Allow uploading an image in JPG, JPEG, or PNG format.

Display:

- Original retinal fundus image.
- Preprocessed image.
- Predicted DR severity.
- Five-class probability chart.
- Estimated image quality.
- Uncertainty indicators.
- Conformal prediction set, if enabled.
- Provisional referable risk.
- Selected triage action.
- Human-readable decision reasons.
- Second-pass inference status.
- Measured processing time.

This is the most important demonstration page.

It must function reliably with genuine trained model checkpoints.

### Module C — Batch Retinal Screening

Support:

- Multiple image uploads.
- Batch progress.
- Per-image outputs.
- Screening queue.
- Sort/filter by DR grade.
- Sort/filter by quality.
- Sort/filter by action.
- Risk prioritization.
- CSV and JSON exports.

The batch view must support the independent review and recapture budgets.

Use defined behavior for overload or infeasible allocations.

### Module D — Research Analytics Center

Display:

- Classification results.
- Confusion matrix.
- Per-grade sensitivity.
- Quadratic weighted kappa.
- Conformal coverage.
- Reliability diagram.
- Risk-coverage curves.
- Quality-severity failure heatmaps.
- Dual-budget trade-offs.
- CPU performance charts.
- Ablation comparisons.

Read these metrics from actual saved experiment outputs.

### Module E — Triage Policy Simulator

This is a major presentation feature.

Allow the researcher to adjust:

- Specialist review budget.
- Recapture budget.
- Severity weighting.
- Uncertainty threshold or policy option.
- Optional compute budget.

Recalculate policy allocation using already computed real model scores and a selected evaluation dataset.

Show how action allocations and empirical screening-risk metrics change.

Clearly separate exploratory simulations from held-out final performance claims.

Display workload trade-offs visually.

Prevent users from treating simulator exploration on test labels as a valid final evaluation.

### Module F — Model Comparison

Compare available genuine trained model variants.

Include:

- Accuracy.
- Macro-F1.
- Quadratic weighted kappa.
- Referable-DR sensitivity.
- Calibration.
- Model size.
- CPU latency.

Do not display model variants that were never evaluated.

### Module G — Research Experiment Explorer

Allow viewing and filtering stored experimental runs.

Show:

- Experiment ID.
- Dataset.
- Seed.
- Configuration.
- Checkpoint.
- Evaluation metrics.
- Generated figures.
- Research notes.

### Module H — Model and System Information

Provide:

- Version identifiers.
- Dataset provenance.
- Supported input formats.
- Operational limitations.
- Preprocessing specifications.
- Evaluation protocol.
- Research disclaimer.
- Deployment information.

## 16. OPTIONAL HIGH-IMPACT FEATURES

Only implement after the core end-to-end prototype works.

Potential enhancements:

- Grad-CAM visualization with explicit limitations.
- Printable research screening report.
- Simple de-identified session history.
- Model comparison overlay.
- Quality-degradation visualization.
- Batch triage ranking visualization.
- ONNX Runtime CPU optimization.
- Retinal crop and preprocessing comparison.
- Presentation/demo mode.
- Exportable experiment reports.

These are secondary to training, triage, experiments, and deployment.

Do not introduce unrelated chatbots or generative AI features.

---

# PART VI — BACKEND AND INFRASTRUCTURE

## 17. TECHNOLOGY STACK

Use the following default technology stack unless real compatibility testing establishes a better option.

**Core language:** Python 3.11 or another verified compatible version.

**Deep learning:** PyTorch and torchvision.

**Image processing:** OpenCV and Pillow.

**Scientific computation:** NumPy, pandas, SciPy, scikit-learn.

**Graphing:** Matplotlib and Plotly.

**Backend:** FastAPI with Pydantic.

**Frontend:** Streamlit.

**Testing:** pytest.

**Code quality:** Ruff and optional mypy.

**Data validation:** Pandera or lightweight explicit validation where justified.

**Optional inference optimization:** ONNX Runtime CPU.

**Experiment tracking:** Local JSON/CSV/YAML files and artifact manifests initially; use more complex tracking tools only when beneficial.

**Persistence:** Filesystem-based artifacts and optional SQLite for non-sensitive local session metadata.

**Deployment:** Docker and Docker Compose.

**Version control:** Git and GitHub.

**CI:** GitHub Actions for quick tests, dependency checks, and linting where available.

Prefer this simple, reliable architecture to unnecessary microservices.

## 18. API DESIGN

Implement these endpoints:

```text
GET  /health
GET  /ready
GET  /model/info

POST /predict
POST /predict/batch

GET  /triage/config
POST /triage/simulate

GET  /research/experiments
GET  /research/metrics

GET  /docs
```

Every endpoint must have typed request and response schemas.

Inference responses should contain:

- Request identifier.
- Model version.
- Predicted DR grade.
- Grade probabilities.
- Referable probability.
- Quality output.
- Uncertainty score.
- Conformal set where available.
- Triage action.
- Decision reason codes.
- Whether additional inference occurred.
- Measured latency.
- Relevant model warnings.

### API quality requirements

Implement:

- Strict file validation.
- Image size limits.
- Robust error responses.
- Batch size limits.
- Model readiness checks.
- Request timeouts where appropriate.
- Exception logging without image contents.
- Resource-efficient model loading.
- OpenAPI documentation.
- Automated endpoint tests.

Ensure missing checkpoints produce an honest unavailable response.

Do not return placeholder predictions.

## 19. DOCKER DEPLOYMENT

Docker support is mandatory.

Provide:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`
- Health checks
- Startup documentation
- Model-path configuration
- CPU-only execution instructions

Expose:

- Streamlit on port `8501`.
- FastAPI on port `8000`.

Expected commands:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Expected application addresses:

```text
http://localhost:8501
http://localhost:8000/docs
```

These commands and endpoints must actually work in the finished project.

### Deployment standards

- Build reproducibly.
- Keep runtime dependencies minimal.
- Avoid downloading large model weights on every startup.
- Load checkpoint files from configured paths.
- Keep datasets outside the runtime image.
- Support Windows with Docker Desktop/WSL2.
- Verify Linux-compatible containers.
- Document RAM, disk, and CPU requirements.
- Support clean shutdown and restart.
- Fail gracefully when assets are unavailable.

Do not claim deployment success based on a valid-looking Dockerfile. Actually build, start, and exercise the application.

---

# PART VII — RESEARCH EVALUATION

## 20. EXPERIMENTAL PROGRAM

Implement the following experiments:

### EXP-01: Basic DR Grading

Train and evaluate the EfficientNet-B0 baseline.

Report grading metrics and real training evidence.

### EXP-02: Quality Assessment

Evaluate image-quality prediction.

Report source-specific quality metrics.

### EXP-03: Uncertainty and Calibration

Evaluate standard confidence, entropy, calibration, and conformal methods.

### EXP-04: Dual-Budget Triage

Evaluate the primary QSART novelty against competing policies.

### EXP-05: Quality × Severity Analysis

Analyze performance across genuine image-quality and severity strata.

### EXP-06: Conditional Inference

Evaluate the effect of selective second-pass inference.

### EXP-07: External Dataset Validation

Evaluate generalization to IDRiD or another appropriately independent dataset.

### EXP-08: Robustness Testing

Evaluate controlled image degradation and distribution shift.

### EXP-09: Component Ablation

Remove quality, severity weighting, uncertainty, separate budget handling, or second-pass inference.

### EXP-10: End-to-End CPU Benchmark

Measure the deployed application on real CPU hardware.

## 21. REQUIRED METRICS

### Disease grading

- Accuracy.
- Macro-F1.
- Per-class precision.
- Per-class recall.
- Confusion matrix.
- Quadratic weighted kappa.
- Referable-DR AUROC and AUPRC.

### Screening and triage

- Missed referable DR per 1,000 screened images.
- Severe-DR miss analysis.
- Automated screening coverage.
- Specialist-review fraction.
- Recapture fraction.
- Empirical risk under fixed budgets.
- Risk-coverage curves.
- Budget violations or infeasibility.
- Results under matched workloads.

Report denominators clearly and distinguish deferred cases from actual specialist-confirmed diagnoses.

### Uncertainty

- Brier score.
- Expected calibration error.
- Empirical conformal coverage.
- Prediction-set size.
- Grade-conditional coverage.

### System performance

- Mean CPU latency.
- P50 latency.
- P95 latency.
- Peak RAM usage.
- Model size.
- Second-pass activation percentage.

### Statistical analysis

Use multiple random seeds where computationally feasible.

Report bootstrap confidence intervals and grouped bootstrap estimates when appropriate identifiers exist.

Avoid unjustified significance claims.

Do not optimize thresholds using the final evaluation data.

## 22. ABLATION TABLE REQUIREMENTS

The experimental system must be capable of producing a table structured approximately as follows:

| Method | Quality-aware | Severity-aware | Two budgets | Conditional CPU | Missed referable DR | Review % | Recapture % |
|---|---|---|---|---|---|---|---|
| Basic baseline | No | No | No | No | Measured | Measured | Measured |
| Confidence policy | No | No | No | No | Measured | Measured | Measured |
| Quality policy | Yes | No | No | No | Measured | Measured | Measured |
| Severity policy | No | Yes | No | No | Measured | Measured | Measured |
| Dual-budget QSART | Yes | Yes | Yes | No | Measured | Measured | Measured |
| Full QSART | Yes | Yes | Yes | Yes | Measured | Measured | Measured |

Replace all placeholders only with actual experimental measurements.

Save exact configurations and seeds behind each reported row.

---

# PART VIII — REPOSITORY DESIGN

## 23. TARGET FILE STRUCTURE

Create or adapt the repository toward this structure:

```text
QSART/
|
|-- README.md
|-- AGENTS.md
|-- MASTER_SPEC.md
|-- TASKS.md
|-- PROGRESS.md
|-- CHANGELOG.md
|
|-- pyproject.toml
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- .dockerignore
|
|-- Dockerfile
|-- docker-compose.yml
|
|-- configs/
|   |-- datasets.yaml
|   |-- model.yaml
|   |-- training.yaml
|   |-- quality.yaml
|   |-- calibration.yaml
|   |-- triage.yaml
|   |-- experiments.yaml
|
|-- data/
|   |-- README.md
|   |-- raw/
|   |-- processed/
|   |-- manifests/
|
|-- src/
|   |-- qsart/
|       |-- data/
|       |-- preprocessing/
|       |-- models/
|       |-- training/
|       |-- calibration/
|       |-- uncertainty/
|       |-- quality/
|       |-- triage/
|       |-- conditional_inference/
|       |-- inference/
|       |-- evaluation/
|       |-- reporting/
|       |-- utils/
|
|-- api/
|   |-- main.py
|   |-- schemas/
|   |-- routes/
|   |-- services/
|
|-- dashboard/
|   |-- app.py
|   |-- pages/
|   |-- components/
|   |-- assets/
|
|-- scripts/
|   |-- inspect_datasets.py
|   |-- prepare_data.py
|   |-- train_dr.py
|   |-- train_quality.py
|   |-- calibrate.py
|   |-- evaluate.py
|   |-- optimize_triage.py
|   |-- run_experiments.py
|   |-- benchmark_cpu.py
|
|-- checkpoints/
|-- experiments/
|-- artifacts/
|-- reports/
|-- notebooks/
|
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- api/
|   |-- e2e/
|
|-- docs/
|   |-- architecture.md
|   |-- dataset_protocol.md
|   |-- training.md
|   |-- research_methodology.md
|   |-- deployment.md
|   |-- api.md
|   |-- limitations.md
|
|-- paper/
|   |-- references.bib
|   |-- literature_matrix.csv
|   |-- experiment_manifest.md
|   |-- figures/
|   |-- tables/
|
|-- .github/
|   |-- workflows/
|       |-- tests.yml
```

Keep the structure extensible without generating unnecessary empty files.

Use Python modules with clean imports and clear interfaces.

Do not duplicate business logic between frontend and backend.

Store trained checkpoints and experiment artifacts in configurable locations.

Do not upload restricted dataset files to GitHub.

---

# PART IX — TWO-WEEK EXECUTION PLAN

## 24. DEVELOPMENT SPRINT

**Sprint Window: October 21 – November 3, 2026**

The objective is to obtain a functioning prototype as quickly as possible, then spend the remaining time strengthening research evidence and deployment quality.

This is an aggressive working schedule, not a claim that all experimental training will necessarily finish within fourteen days.

### Days 1–2: Foundation and Real-Image Baseline

**October 21–22**

- Audit any existing QSART code.
- Reuse working policy, tests, and Docker scaffolding where appropriate.
- Establish repository and environment.
- Verify APTOS download and label access.
- Implement real preprocessing.
- Begin or run baseline EfficientNet-B0 training.
- Produce an initial genuine checkpoint where feasible.
- Demonstrate model-backed predictions.
- Prepare Review-1 evidence.

**Exit condition:** Valid retinal images can flow through a real inference module, or the exact training blocker is documented.

### Days 3–4: Quality and Uncertainty

**October 23–24**

- Integrate available DeepDRiD annotations.
- Implement the quality model.
- Train or evaluate quality inference.
- Add calibrated confidence and entropy.
- Implement the initial conformal evaluation.
- Write model tests and output schemas.

**Exit condition:** Unified disease, quality, and uncertainty outputs are working.

### Days 5–6: Core Triage and Backend

**October 25–26**

- Implement all three screening actions.
- Implement dual-budget allocation.
- Add severity weighting.
- Add workload simulation.
- Implement FastAPI.
- Connect the complete inference pipeline.
- Test single-image API predictions.

**Exit condition:** Real images can produce full screening decisions through the backend.

### Day 7: First Complete Deployment

**October 27**

- Create or finish Streamlit's essential pages.
- Connect frontend to API.
- Dockerize the system.
- Test a complete image-to-action workflow.
- Fix critical integration defects.

**Milestone: QSART v0.1 — Working End-to-End Prototype.**

This milestone takes precedence over advanced experimental features.

### Days 8–9: Research Novelties

**October 28–29**

- Finish competing triage policy baselines.
- Add ordinal conformal prediction if not already complete.
- Implement severity-aware quality analysis.
- Implement conditional second-pass inference.
- Add configurable policy simulation.

**Exit condition:** Each novelty has an executable implementation and a documented test strategy.

### Days 10–11: Experimental Evaluation

**October 30–31**

- Run comparative triage evaluations.
- Run degradation experiments.
- Run calibration analysis.
- Measure CPU performance.
- Run external evaluation where data is accessible.
- Generate tables and figures.

**Exit condition:** Reproducible experimental artifacts exist with genuine measurements.

### Days 12–13: Product Enhancement and Reliability

**November 1–2**

- Improve dashboard UX.
- Add batch workflows.
- Add policy and experiment visualizations.
- Run ablations.
- Optimize memory and performance.
- Strengthen error handling.
- Run complete integration and Docker tests.

**Exit condition:** The application is stable enough for an independent demonstration.

### Day 14: Release Candidate

**November 3**

- Freeze a tested checkpoint and configuration.
- Rebuild Docker from a clean environment.
- Execute final smoke tests.
- Generate a project demo walkthrough.
- Export final experiment summaries.
- Audit incomplete items.
- Tag a release candidate if acceptance tests pass.

**Milestone: QSART v1.0 Release Candidate.**

If any mandatory acceptance criterion remains unmet, label the release as partial rather than declaring full completion.

---

# PART X — PRIORITY CONTROL

## 25. DEVELOPMENT PRIORITY SYSTEM

Every task must be classified as one of three levels.

### P0 — Mandatory

These determine whether the project is genuinely working.

- Real dataset pipeline.
- Trained DR model.
- Quality module.
- Model-backed inference.
- Three-action triage.
- Dual-budget policy.
- FastAPI.
- Streamlit upload-and-analysis page.
- Docker deployment.
- End-to-end tests.
- Actual research experiment outputs.

P0 tasks receive priority over UI decoration and optional infrastructure.

### P1 — Essential for Research Quality

- Ordinal conformal evaluation.
- Calibration comparison.
- Robustness and severity analysis.
- Conditional inference.
- Baseline comparisons.
- Ablations.
- Confidence intervals.
- External evaluation.
- Research experiment explorer.

### P2 — Product Excellence

- Highly polished dashboard.
- Grad-CAM.
- Exportable reports.
- Advanced analytics filters.
- CPU optimization through ONNX.
- Additional dataset integrations.
- Enhanced visual explanations.
- Advanced deployment conveniences.

P2 tasks must never jeopardize P0 completion.

## 26. PARALLEL DEVELOPMENT

Use parallel workstreams when tools and environment permit:

**Workstream A — Machine Learning**

Datasets, model training, quality estimation, calibration.

**Workstream B — Research Algorithms**

Triage, conformal prediction, conditional inference, experimental baselines.

**Workstream C — Product**

Backend, frontend, batch screening, analytics.

**Workstream D — Verification and Deployment**

Testing, reproducibility, Docker, documentation.

Merge changes incrementally.

Do not permit multiple agents to overwrite the same files without coordination.

Maintain one authoritative specification and backlog.

---

# PART XI — CODE QUALITY AND VERIFICATION

## 27. ENGINEERING STANDARDS

Code must be:

- Modular.
- Typed where useful.
- Configurable.
- Reproducible.
- Testable.
- Consistent.
- Documented.
- Robust to invalid inputs.
- Efficient on CPU.
- Suitable for Git version control.

Avoid giant notebook-only implementations.

Notebooks are permitted for exploratory research, but reusable implementations belong in Python modules.

Pin compatible dependencies.

Avoid unnecessary technologies.

Preserve existing functioning code.

## 28. MANDATORY TESTS

### Unit tests

Cover:

- Image preprocessing.
- Label mapping.
- Model outputs.
- Probability normalization.
- Calibration calculations.
- Conformal prediction set construction.
- Severity risk scoring.
- Triage allocation.
- Budget accounting.
- Conditional inference activation.

### Integration tests

Cover:

- Dataset reader → model.
- Model → calibration.
- Calibration → triage.
- Inference pipeline → API.
- API → frontend.
- Model loading → prediction.
- Experiment configuration → saved metrics.

### System tests

Cover:

- Single-image upload.
- Batch upload.
- Invalid image format.
- Missing checkpoint.
- Zero-budget cases.
- Overloaded triage policies.
- Docker startup.
- CPU-only inference.
- Result export.

Use synthetic fixtures for software tests.

Use permitted real retinal images for final model-backed integration tests.

## 29. DEFINITION OF DONE

A feature is only marked complete when:

1. The implementation exists.
2. Required dependencies are available.
3. Tests have been executed.
4. Relevant tests pass.
5. The feature integrates with the rest of the system.
6. The output has been inspected.
7. Documentation is updated.
8. No critical known failure remains hidden.

Merely generating code is not completion.

---

# PART XII — RESEARCH PAPER PREPARATION

## 30. PAPER DEVELOPMENT POLICY

**The formal research paper will be written after the core project implementation and major experiments are complete.**

However, preparation of the research evidence begins immediately.

Throughout implementation, preserve:

- Literature notes.
- Bibliographic entries.
- Research hypotheses.
- Dataset protocols.
- Training configurations.
- Baseline definitions.
- Triage objective functions.
- Experimental procedures.
- Figures and tables.
- Statistical analysis.
- Failure cases.
- Interpretation notes.
- Limitations.

Do not spend the early development sprint polishing an abstract for results that have not been measured.

### Literature requirement

Maintain at least 20 verified relevant research references, prioritizing recent work from 2023–2026.

Maintain the existing top-six literature comparison, including the verified selected base paper and relevant work on retinal quality assessment, retinal foundation models, selective classification, abstention, and prospective screening.

Recheck paper identity, publication details, and claimed results before citing.

### Paper outline

After the main experiments are completed, prepare:

1. Abstract.
2. Introduction.
3. Related Work.
4. Research Gap and Contributions.
5. Proposed Methodology.
6. Mathematical Formulation.
7. Datasets and Experimental Setup.
8. Results.
9. Ablation Studies.
10. Quality and Domain-Shift Analysis.
11. Efficiency Analysis.
12. Discussion.
13. Limitations.
14. Conclusion.
15. References.

Use an appropriate IEEE-style template if that matches the target venue.

Do not guarantee acceptance at a conference or journal.

Research claims must be aligned with the strength of the actual results.

---

# PART XIII — LIVING SPECIFICATION AND CHANGE MANAGEMENT

## 31. MASTER PROMPT VERSIONING

This specification will be refined over a two-week development period.

Treat this document as:

`MASTER_SPEC_v1.0`

Maintain a stable current version at:

`MASTER_SPEC.md`

Record updates in:

`CHANGELOG.md`

Each meaningful change should record:

- Version number.
- Date.
- Changed requirement.
- Reason for the change.
- Affected modules.
- Required tests.
- Research implications.
- Compatibility with existing work.

### Protected project decisions

Do not casually change these without an explicit architectural or research decision:

- Retinal fundus imaging modality.
- Diabetic retinopathy screening focus.
- Three selected scientific contributions.
- Independent review and recapture budgets.
- Three screening action categories.
- CPU-first deployment.
- Docker requirement.
- Zero additional cost.
- Research evidence integrity.
- Formal paper writing after substantive implementation.

### Flexible engineering decisions

These may be refined based on real evidence:

- Backbone architecture.
- Input resolution.
- Augmentation strategy.
- Quality model architecture.
- Calibration implementation.
- Optimization method.
- Conditional inference technique.
- Dashboard design.
- Storage strategy.
- Experiment tracking approach.
- Additional datasets.

All changes must preserve working end-to-end functionality.

---

# PART XIV — REPORTING AND AGENT EXECUTION RULES

## 32. HOW TO REPORT PROGRESS

At the end of each meaningful development session, produce a concise engineering progress report.

Use this format:

**QSART Development Status**

- Current specification version.
- Latest verified commit.
- Current milestone.
- Components implemented.
- Components actually tested.
- Test results.
- Real experimental metrics generated.
- Docker verification status.
- Known blockers.
- Files modified.
- Next highest-priority executable tasks.

Use the following completion labels:

- `VERIFIED COMPLETE`
- `IMPLEMENTED — UNVERIFIED`
- `IN PROGRESS`
- `BLOCKED`
- `NOT STARTED`

Do not use completion labels inaccurately.

## 33. AGENT BEHAVIOR

Follow these instructions:

**Instruction A:** Inspect the existing repository before making changes.

**Instruction B:** Begin implementing once the initial audit is complete. Avoid repetitive planning cycles.

**Instruction C:** Make reasonable engineering decisions independently.

**Instruction D:** Run code and test it wherever execution tools are available.

**Instruction E:** When a failure occurs, inspect logs and fix the underlying cause.

**Instruction F:** Never fake successful training, inference, testing, or deployment.

**Instruction G:** Keep documentation synchronized with the implementation.

**Instruction H:** Prioritize working components over unnecessary abstractions.

**Instruction I:** Continue with the highest-priority unblocked task during the available execution session.

**Instruction J:** Preserve reproducibility and research validity.

**Instruction K:** Ask for human intervention only when genuinely necessary, such as permissions, unavailable dataset access, or decisions with major scientific consequences.

**Instruction L:** Do not disclose private data or commit medical images without permission.

**Instruction M:** Do not claim specialist interpretation or clinical validation has occurred merely because the application produces a screening recommendation.

**Instruction N:** Review the full dependency chain after major architectural changes.

**Instruction O:** Always identify the first broken or incomplete part of the real-image-to-decision pipeline before adding secondary features.

---

# PART XV — FINAL DELIVERABLES

## 34. REQUIRED FINAL PROJECT OUTPUT

Deliver a complete QSART repository containing:

### Working AI System

- Trained DR grading model.
- Image-quality assessment.
- Calibration and uncertainty.
- Conformal prediction.
- Severity-aware dual-budget triage.
- Conditional inference.
- CPU-compatible inference pipeline.

### Working Software

- FastAPI backend.
- Streamlit dashboard.
- Single-image screening.
- Batch screening.
- Policy simulator.
- Research analytics.
- Export functionality.
- Docker deployment.

### Research Evidence

- Measured grading performance.
- Quality assessment performance.
- Calibration analysis.
- Triage baseline comparisons.
- Risk-workload trade-offs.
- Severity-specific failure analysis.
- Conditional inference benchmarks.
- Ablation studies.
- External evaluation, where suitable data is accessible.
- Confidence intervals and limitations.

### Engineering Evidence

- Passing automated tests.
- Deployment verification.
- Recorded CPU benchmarks.
- Configuration files.
- Model metadata.
- Reproducible scripts.
- Setup documentation.
- Architecture documentation.

### Academic Evidence

- Research objectives.
- Literature matrix.
- Mathematical methodology.
- Experimental protocols.
- Figures and tables.
- Paper-ready technical findings.
- Final demonstration material.

## 35. FINAL DEMONSTRATION

The required final demonstration follows this workflow:

**Step 1:** Open the deployed QSART dashboard.

**Step 2:** Upload a real permitted retinal fundus image.

**Step 3:** Validate and preprocess the image.

**Step 4:** Perform actual model inference.

**Step 5:** Display predicted DR severity and probabilities.

**Step 6:** Evaluate image quality.

**Step 7:** Compute uncertainty and conformal outputs.

**Step 8:** Execute conditional inference where applicable.

**Step 9:** Apply the severity-aware dual-budget triage policy.

**Step 10:** Display the selected screening action and reasons.

**Step 11:** Export the screening result.

**Step 12:** Open the research dashboard and show genuine evaluations supporting the system.

This full sequence must be executable.

---

# PART XVI — YOUR FIRST ACTIONS

## 36. START IMPLEMENTATION

Begin by completing the following actions in order:

1. Audit existing QSART project files, notebooks, documents, models, and repositories.
2. Locate and preserve useful prior implementation, including policy code, metadata auditing, test scaffolding, and Docker scaffolding.
3. Create or update `MASTER_SPEC.md`, `AGENTS.md`, `TASKS.md`, and `PROGRESS.md`.
4. Verify local environment, Python, PyTorch, Docker, and available compute resources.
5. Confirm legally accessible APTOS and DeepDRiD data.
6. Implement and validate the real-image preprocessing pipeline.
7. Train or fine-tune the first lightweight DR classification model.
8. Save a real trained checkpoint.
9. Run CPU inference on a real retinal image.
10. Build the minimum functional FastAPI and Streamlit workflow.
11. Implement the initial three-action triage system.
12. Package and test the first working Docker prototype.
13. Continue implementing the research contributions and experiments.
14. Refine the dashboard and research system using the remaining sprint capacity.

### First checkpoint

The earliest important achievement is:

**A user uploads a real retinal image to the application, a genuinely trained model processes it, and the system produces an honest, interpretable, evidence-backed screening output.**

Reach this before investing heavily in cosmetic features.

### Final instruction

Build QSART as both an impressive software engineering project and a serious M.Tech research contribution.

Be ambitious in the final architecture, disciplined in execution, and conservative in scientific claims.

**Never sacrifice a working prototype for an impressive but incomplete design. Never sacrifice scientific validity for attractive experimental numbers.**

Success requires three things together:

**A genuinely trained AI system + a reliable deployable application + reproducible experimental evidence.**

Start with the existing project audit and proceed to implementation immediately.
