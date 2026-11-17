# QSART-X — MASTER DEVELOPMENT PROMPT
## Complete AI Retinal Screening Research Platform, Website, Full-Stack Application and Publication Pipeline

### 1. Your role

Act as a multidisciplinary engineering and research team consisting of:

- Senior AI/ML Research Scientist
- Medical Computer Vision Engineer
- Full-Stack Software Architect
- Senior React UI/UX Engineer
- Backend and Database Engineer
- MLOps and Docker Engineer
- Experimental Research and Scientific Writing Specialist
- Software Testing and Security Engineer

Your mission is to implement QSART-X as a complete, reproducible, research-oriented AI application with a sophisticated website and a real working backend.

Do not create only a beautiful frontend or a static prototype. Build the complete system in stages, including dataset processing, model training, inference, triage, a connected web application, experiments, and deployment.

Make practical implementation decisions independently, record those decisions, and continuously test your work.

---

### 2. Project identity

**Core research title:**

QSART: Quality- and Severity-Aware Dual-Budget Triage for Reliable Diabetic Retinopathy Screening

**Software platform name:** QSART-X

**Extended research direction:**

Budget-Constrained Adaptive Multi-View Acquisition and Safety-Aware Triage for Diabetic Retinopathy Screening

**Academic context:**
M.Tech in Artificial Intelligence and Machine Learning, VIT Vellore.

**Primary objectives:**
1. Build a complete working medical-AI research prototype.
2. Develop a defensible, experimentally measurable research contribution.
3. Produce reproducible results suitable for a peer-reviewed manuscript.
4. Build a professional full-stack website and screening application.
5. Support future extensions without major architectural changes.
6. Achieve local CPU-first deployment through Docker.

This project is a retrospective research and screening-decision-support prototype. It is not a clinically validated or approved autonomous diagnostic system.

### 3. Non-negotiable constraints

- Use free and open-source development tools.
- Work on a mid-range Windows laptop, with CPU support and optional GPU acceleration.
- Do not require paid APIs, paid cloud services or commercial model subscriptions.
- Make the application deployable with Docker Compose.
- Use real datasets and actual inference for research results.
- Do not fabricate accuracy, clinical metrics, improvements, references or experimental outcomes.
- Do not claim novelty merely from combining established methods.
- Do not claim guaranteed paper acceptance or medical safety.
- Do not assume real recapture outcomes can be inferred from unrelated images.
- Never expose patient images or private records on the public website.
- Support incremental implementation and testing.
- Ensure the whole application can run locally.

Use synthetic demonstration data only where clearly identified as DEMO DATA. Never include such results in research evaluations.

---

# PART A — PROFESSIONAL WEBSITE

### 4. Build the QSART-X website

Create a complete, modern, polished medical-technology website.

**Design requirements:**

- Professional healthcare-research SaaS appearance.
- Clean, sophisticated, minimalist design.
- Primary colors: deep navy, medical teal, white and restrained cyan accents.
- Excellent typography, spacing and visual hierarchy.
- Responsive desktop, tablet and mobile layouts.
- Smooth but restrained motion.
- Accessible components, keyboard navigation and adequate contrast.
- Light mode by default; optional dark mode for the application.
- No generic-looking AI-generated layouts.
- No intrusive animations or meaningless decorative elements.
- Consistent design system using reusable components.

Use React, TypeScript, Vite, Tailwind CSS, and appropriate accessible UI components.

### 5. Public website pages

**Home page**

Build a compelling landing page featuring:

- QSART-X identity and logo.
- Clear explanation of the research problem.
- Interactive visual explaining retinal screening.
- Explanation of the three screening actions.
- Research contribution overview.
- Dataset and methodology highlights.
- Real, measured performance metrics only when available.
- Demonstration entry point.
- Research documentation access.
- Project status and roadmap.
- Team and institutional affiliation where accurate.

Hero concept:

"Smarter Retinal Screening. Safer Decisions. Responsible AI."

Include suitable illustrations or licensed research imagery. Never show identifiable patient data without authorization.

**About page**

Explain diabetic retinopathy, retinal fundus photography, screening limitations, unreliable images and the motivation for risk-aware AI triage.

**Research page**

Explain the research questions, hypotheses, methodology, proposed novelty, limitations, baselines, datasets and evaluation plan.

**Architecture page**

Provide an interactive architecture diagram showing:

Fundus Input → Image Processing → Quality Estimation + Severity Estimation + Uncertainty Estimation → Decision Engine → Screening Action.

Include the planned second-view acquisition extension.

**Publications page**

Provide verified references, DOI links, research reports and eventually our manuscript. Label unpublished work clearly.

**Demo page**

Provide access to a research demonstration. If the model is not trained, explain that inference is unavailable instead of producing invented predictions.

**Contact / Team page**

Provide a professional project contact interface, team details and appropriate acknowledgements.

**Responsible AI / Privacy page**

Explain limitations, data handling, research-only status and appropriate use.

The public website must remain useful without running the ML inference service.

---

# PART B — RETINAL SCREENING APPLICATION

### 6. Build a secure screening portal

Create an authenticated application with separate roles:

- Research administrator
- Research operator
- Clinician/reviewer
- Read-only research viewer

Use role-based access controls and secure authentication.

### 7. Screening dashboard

Display:

- Number of screening cases processed.
- Case status distribution.
- Model availability and version.
- Image-quality distribution.
- Severity prediction distribution.
- Specialist-review queue.
- Recapture/second-view requests.
- Resource-budget utilization.
- Inference latency.
- Research evaluation summaries.

Clearly distinguish real processing records, validation results and demonstration records.

### 8. Image upload and analysis

Support:

- JPG/PNG retinal image upload.
- Basic format and size validation.
- Image orientation and preprocessing.
- Image quality assessment.
- Five-grade DR severity prediction.
- Class probabilities.
- Referable DR risk.
- Severe DR risk.
- Confidence and uncertainty indicators.
- Triage recommendation.
- Model version and inference timestamp.

Display the retinal image alongside its analysis.

Avoid presenting model estimates as definitive medical diagnoses.

### 9. Three-action triage system

Implement the previously agreed QSART actions:

**Action A0 — Automated non-referable screening disposition**

Available only when risk estimates satisfy predefined screening criteria. The UI must distinguish automated triage from clinical diagnosis.

**Action A1 — Recapture / additional image acquisition**

Recommend another image when identifiable quality deficiencies or expected information benefit justify acquisition.

**Action A2 — Specialist review**

Escalate high-risk, ambiguous, severe, insufficient-quality or unresolved cases.

Implement separate resource constraints:

- B_recapture: available imaging/recapture capacity.
- B_review: available specialist-review capacity.

The decision algorithm must account for these two constraints separately.

High estimated disease severity must receive appropriate risk weighting. Avoid strategies that automatically clear a case merely because the model is confident.

### 10. Clinician review workspace

Implement:

- Case listing and filtering.
- Prioritized review queue.
- Image inspection.
- Prediction and uncertainty display.
- Optional second-view comparison.
- Reviewer assessment.
- Model-versus-reviewer disagreement indicators.
- Case notes.
- Human override.
- Action history and audit log.
- Research report generation.

Reviewer outcomes must be recorded as user-entered assessments, not treated automatically as independently verified reference labels.

### 11. Reports

Generate structured screening-research reports containing:

- Case identifier.
- Image-quality status.
- Model outputs.
- Suggested workflow action.
- Limitations and uncertainty.
- Reviewer notes, if available.
- Model version.
- Timestamp.
- Research-only notice.

Provide PDF export and a printer-friendly view.

---

# PART C — DATASETS AND DATA ENGINEERING

### 12. Required datasets

Use these primary datasets:

**APTOS 2019**
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Purpose: Five-grade diabetic retinopathy model development.

**DeepDRiD**
https://github.com/deepdrdoc/DeepDRiD

Purpose: Image-quality assessment, severity evaluation, paired-view research and adaptive-acquisition experiments.

Focus on regular fundus images initially.

**IDRiD**
https://idrid.grand-challenge.org/Data/

Purpose: Independent external severity and reliability evaluation.

**EyeQ — optional extension**
https://github.com/HzFu/EyeQ

Purpose: Additional image-quality modeling, subject to verified image availability and usage conditions.

### 13. Dataset management

Implement:

- Source and license manifests.
- Dataset integrity validation.
- Image-label matching.
- Duplicate detection.
- Patient-disjoint splits wherever identifiers permit.
- Class-distribution reports.
- Image-quality summaries.
- Reproducible preprocessing.
- Dataset versioning and checksums.
- Protected external test partitions.
- Leakage checks across datasets.

Never assume that image IDs represent unique patients.

Never train on external test data or tune thresholds after observing held-out test outcomes.

DeepDRiD paired views must be associated with the correct patient and eye.

These paired views represent different imaging views, not necessarily a failed photograph followed by a successful retake.

Allow manual dataset download and local import where account authentication or usage conditions require it.

Provide clear instructions when a source cannot be downloaded automatically.

---

# PART D — MACHINE LEARNING ENGINE

### 14. Baseline severity model

Develop a five-class diabetic retinopathy classifier.

Use EfficientNet-B0 as the primary baseline, with MobileNetV3 as an efficiency comparison.

Severity grades:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

Operational research definitions:

- Referable DR: grade ≥ 2
- Severe DR: grade ≥ 3

These are study definitions, not universally sufficient clinical referral criteria.

Do not infer diabetic macular edema solely from DR grade.

Train using reproducible preprocessing, seeded experiments, class-aware evaluation and an appropriately separated validation set.

### 15. Image-quality module

Estimate gradability and meaningful failure types where suitable annotations are available.

Potential quality features:

- Blur and clarity.
- Illumination.
- Contrast.
- Field definition.
- Cropping.
- Artifacts.

Distinguish learned predictions from heuristic measurements.

Test how quality failures affect different disease-severity groups.

### 16. Uncertainty estimation

Implement and compare:

- Maximum softmax probability.
- Predictive entropy.
- Confidence margin.
- Calibration methods.
- Ordinal conformal prediction, where feasible.
- Selective prediction.

Report calibration performance and coverage limitations, particularly under dataset shift.

Do not assume in-distribution calibration guarantees out-of-distribution safety.

---

# PART E — PRIMARY RESEARCH NOVELTY

### 17. Severity-aware dual-budget decision optimization

Research a calibrated decision policy that considers:

- Severity probabilities.
- Estimated referable DR risk.
- Estimated severe DR risk.
- Image quality.
- Model uncertainty.
- Specialist-review capacity.
- Imaging-acquisition capacity.
- Costs of different decision errors.

The primary optimization target is to reduce severity-weighted missed referral outcomes while respecting workload limits.

At matched budgets, compare against:

1. Classifier-only screening.
2. Confidence-threshold triage.
3. Entropy-based abstention.
4. Quality-only triage.
5. Calibrated uncertainty-based triage.
6. Tuned quality-plus-uncertainty triage.
7. Proposed QSART decision optimization.

Include explicit ablations for severity weighting, quality, calibration and separate budgets.

All comparisons must use clearly specified and reproducible assumptions about action outcomes.

### 18. Stronger research extension: Value of Information

Develop QSART-VI as an experimental module within QSART-X.

Research question:

Can the system estimate when another retinal view is sufficiently useful to justify acquisition, rather than always using a fixed number of images?

Develop an expected-benefit estimation method using:

- Current predictive uncertainty.
- Current image-quality indicators.
- Risk of missed referable or severe DR.
- Potential information gain from a second view.
- Acquisition cost.
- Specialist-review cost.

Compare:

- First view only.
- Always acquire both views.
- Quality-based second-view acquisition.
- Uncertainty-based second-view acquisition.
- Proposed value-of-information policy.

Use DeepDRiD paired views for retrospective experiments.

When performing a simulated acquisition decision, ensure the policy only accesses information available before that decision. Do not leak the hidden second-view label or features into the first-view decision.

Explicitly separate observed second-view improvements from assumed benefits of a new photograph.

This is a proposed research contribution and must be evaluated rather than declared novel in advance.

### 19. Domain-shift and degradation research

Evaluate reliability under:

- Blur.
- Illumination change.
- Contrast changes.
- Image compression.
- Field-of-view truncation.
- External dataset shift.

Compare performance across disease severity.

Report both successful and unsuccessful stress tests.

Do not claim robustness or performance guarantees from synthetic corruptions alone.

### 20. CPU-efficient selective inference

Develop optional conditional second-pass inference.

Process all images using a lightweight model initially.

Invoke a second model or additional image-processing stage only when its expected decision benefit justifies the added computation.

Measure:

- Mean inference latency.
- Median and 95th-percentile latency.
- Peak memory.
- Second-pass activation fraction.
- Prediction quality.
- Screening risk at matched workloads.

This optimization is optional until the baseline and main novelty work reliably.

---

# PART F — RESEARCH ANALYTICS LABORATORY

### 21. Build an integrated experiment dashboard

Create a dedicated Research Lab section in the website.

Support viewing and comparing actual experiment outputs:

- Dataset statistics.
- Confusion matrices.
- ROC and precision-recall curves.
- Calibration curves.
- Reliability diagrams.
- Severity-specific error rates.
- Risk-versus-workload curves.
- Recapture-versus-review trade-offs.
- Domain-shift comparisons.
- Model inference latency.
- Ablation studies.
- Confidence intervals.
- Budget-sensitivity analyses.

Include an interactive budget explorer:

Users can adjust B_review and B_recapture to see the outcome of a previously validated policy or explicitly labeled retrospective simulation.

Do not invent results when suitable experimental data are unavailable.

Track experiment configuration, seeds, checkpoints, dataset versions and timestamps.

### 22. Primary scientific metrics

Report:

- Macro-F1.
- Sensitivity and specificity.
- AUROC for referable DR.
- Quadratic weighted kappa.
- Confusion matrix.
- Calibration error and Brier score.
- Missed referable DR cases per 1,000 screened.
- Missed severe DR cases per 1,000 screened.
- Severity-weighted missed-case risk.
- Review and acquisition workload.
- Coverage and selective risk.
- CPU inference time.
- Peak memory.
- Bootstrap 95% confidence intervals.

Measure missed cases over a clearly defined screening population.

Report how human review and second-view outcomes are modeled, rather than assuming every referral or recapture resolves disease risk.

Use appropriate statistical tests and acknowledge small-sample uncertainty.

### 23. Publication support

Create a reproducible research package including:

- Hypotheses.
- Verified literature matrix.
- Dataset provenance.
- Methodology documentation.
- Mathematical formulation.
- Baseline descriptions.
- Experiment scripts.
- Statistical analysis.
- Tables and figures.
- Ablation results.
- Limitations.
- Reproducibility checklist.

Support export to IEEE-compatible tables, figures and LaTeX.

Maintain a bibliography of up to 50 relevant verified research papers, prioritizing reputable peer-reviewed sources.

Never invent citations, DOIs, author names, publication status or experimental improvements.

Manuscript drafting should begin from verified experimental results and distinguish the original QSART contribution from later QSART-X extensions.

---

# PART G — ARCHITECTURE AND SECURITY

### 24. Technology choices

Frontend:
- React
- TypeScript
- Vite
- Tailwind CSS
- Accessible reusable UI components
- Recharts and Plotly

Backend:
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL
- Alembic migrations

Machine learning:
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- SciPy
- ONNX Runtime when exported models are validated

Research:
- pandas
- NumPy
- Jupyter
- MLflow
- Matplotlib
- pytest

Deployment:
- Docker
- Docker Compose
- CPU-compatible inference containers

Use actively maintained stable versions that are compatible with one another. Pin dependencies and document the environment.

### 25. Backend API

Implement documented and validated endpoints for:

- Authentication.
- Dataset metadata.
- Retinal image upload.
- Image-quality assessment.
- DR severity prediction.
- Risk estimation.
- Triage policy execution.
- Review queue.
- Case details.
- Reviewer actions.
- Research experiments.
- Model registry.
- Monitoring.
- Report generation.
- Health and readiness checks.

Use structured error responses, typed schemas and OpenAPI documentation.

### 26. Database and storage

Store:

- Users and roles.
- Screening case metadata.
- Image references.
- Quality predictions.
- Severity predictions.
- Uncertainty estimates.
- Policy decisions.
- Model and policy versions.
- Review actions.
- Audit logs.
- Experiment metadata.
- Reports.

Use local filesystem storage or a suitable self-hosted object-storage abstraction for image files.

Do not store large retinal-image binaries directly in normal database rows without justification.

### 27. Security and safeguards

Implement:

- Secure password handling.
- Role-based authorization.
- Input validation.
- File-size and upload checks.
- Safe image processing.
- Configuration through environment variables.
- No hardcoded production secrets.
- Database migrations.
- Restricted data access.
- Log redaction.
- Audit trails.
- Clear research-use disclaimers.
- Model-unavailable fallback behavior.

Human reviewers must be able to override proposed actions.

Never claim the system is ready for autonomous real-patient clinical deployment.

---

# PART H — REPOSITORY AND DOCKER DELIVERY

### 28. Project structure

Use a clean monorepo:

qsart-x/
- apps/web/
- services/api/
- services/ml/
- packages/shared/
- data/raw/
- data/processed/
- data/manifests/
- models/
- experiments/
- notebooks/
- research/
- docs/
- tests/
- infrastructure/docker/
- scripts/

Separate model training from application inference.

Build services for:

- Frontend.
- API.
- PostgreSQL.
- Optional experiment tracking.

Include environment examples, database migrations, health checks and persistent volumes.

Ensure Docker containers do not require GPU hardware.

### 29. Required developer experience

Provide:

- Complete README.
- Windows PowerShell setup instructions.
- Python environment setup.
- Dataset download/import guide.
- Dependency installation.
- Model-training commands.
- Evaluation commands.
- Backend startup.
- Frontend startup.
- Docker Compose startup.
- Testing instructions.
- Troubleshooting instructions.
- Dataset-license and attribution notes.
- Example environment configuration.

Public website pages may be built as a static production bundle. The connected screening application requires an operational backend.

Do not falsely describe GitHub Pages as capable of hosting the FastAPI and PostgreSQL services.

---

# PART I — IMPLEMENTATION STRATEGY

### 30. Build in working milestones

**Milestone 1 — Repository and website**

- Create the monorepo.
- Establish frontend and backend.
- Implement the design system.
- Build public website pages.
- Implement navigation and responsive layout.
- Connect frontend to a real API health endpoint.
- Configure database and Docker.
- Test that the website works.

**Milestone 2 — Dataset foundation**

- Implement dataset discovery/import.
- Verify files and annotations.
- Generate dataset manifests.
- Add preprocessing.
- Create valid splits.
- Generate dataset audit reports.

**Milestone 3 — Working AI inference**

- Implement EfficientNet-B0 baseline.
- Develop training and evaluation scripts.
- Save reproducible checkpoints.
- Build model-loading and inference API.
- Connect image upload to real predictions.
- Show unavailable states when no checkpoint is installed.

**Milestone 4 — QSART core**

- Add image-quality assessment.
- Add uncertainty estimation.
- Implement dual-budget triage.
- Build screening and reviewer interfaces.
- Add audit logs and case reporting.

**Milestone 5 — Research validation**

- Implement strong baselines.
- Perform matched-budget tests.
- Evaluate severity-specific misses.
- Run degradation and external tests.
- Generate statistical analysis and research figures.

**Milestone 6 — QSART-X research extension**

- Implement paired-view analysis.
- Develop value-of-information models.
- Compare adaptive acquisition policies.
- Add the budget simulator to the dashboard.
- Document methodological limitations.

**Milestone 7 — Production-quality research release**

- Improve reliability and security.
- Optimize CPU inference.
- Add automated tests.
- Package Docker deployment.
- Validate end-to-end workflows.
- Finalize research documentation.
- Prepare manuscript materials.

### 31. Acceptance criteria

The project is ready for a research demonstration only when:

1. The frontend and backend run successfully.
2. The website is responsive and complete.
3. Real images can be uploaded securely.
4. A valid model checkpoint produces actual predictions.
5. Image quality and uncertainty results are visible.
6. The triage engine executes correctly with both budget limits.
7. Reviewer actions persist in the database.
8. Research experiment results can be reproduced.
9. External evaluation is separated from training.
10. Docker Compose starts the required services.
11. Automated tests cover critical workflows.
12. The application displays accurate safety and research limitations.
13. All reported manuscript results can be traced to experiment outputs.

Do not declare completion simply because the interface looks finished.

---

# PART J — HOW TO EXECUTE THIS TASK

### 32. Instructions to the development agent

First inspect the current repository and preserve all useful existing work.

If the repository already contains a QSART implementation, extend and refactor it instead of rebuilding everything unnecessarily.

Then:

1. Produce a concise architecture and implementation roadmap.
2. Identify current code, datasets, checkpoints and missing dependencies.
3. Begin with Milestone 1.
4. Implement actual source files, not pseudocode.
5. Run tests and inspect errors after each milestone.
6. Fix failures before expanding the feature scope.
7. Keep a project progress file tracking completed, pending and blocked work.
8. Record major architectural decisions.
9. Use separate research and production configuration.
10. Verify API/frontend integration.
11. Never fabricate experimental outputs.
12. Keep the application runnable after every milestone.

If dataset downloads require my Kaggle account, manual acceptance of terms, or another external permission, provide exact instructions and continue with independent development tasks.

If training is too computationally expensive locally, implement the complete pipeline with a smaller reproducible baseline first and document the remaining compute requirements.

Prioritize working functionality, scientific validity, modular design and maintainability over superficial feature count.

Do not stop after generating a plan. Proceed to implementation using the tools available in the coding environment.

### Final objective

Deliver QSART-X as a polished, full-stack retinal-screening research platform with:

- A professional public website.
- A functioning AI inference workflow.
- Quality, severity and uncertainty estimation.
- A tested dual-budget triage engine.
- A clinician-oriented review portal.
- A reproducible experiment laboratory.
- A defensible adaptive-acquisition research extension.
- CPU-compatible Docker deployment.
- Traceable research results suitable for scientific manuscript preparation.

**Build a credible research system, not merely an impressive demonstration.**