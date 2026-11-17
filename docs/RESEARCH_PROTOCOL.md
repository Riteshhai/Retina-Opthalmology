# QSART Pre-specified Research Protocol — Version 0.1 (2026-10-18)

## Target
Retrospective, nonclinical investigation into the decision risk of low-quality retinal images for detecting diabetic retinopathy. Demonstrate a fully offline research prototype on CPU and write a peer-reviewed manuscript, if results support it. This is **not** an autonomous clinical screening device.

## Competitive novelty audit
- Pan et al. 2026 QGDR models quality-guided DR and shows quality-severity coupling (https://pmc.ncbi.nlm.nih.gov/articles/PMC13259753/).
- Pinto et al. 2025 NaIA-RD already combines DR + gradability, with nonreferable/referable/ungradable output and real-world workflow (https://pmc.ncbi.nlm.nih.gov/articles/PMC12222278/).
- Feng et al. 2026 externally validates quality-aware ordinal grading and selective referral (https://doi.org/10.1016/j.bspc.2026.111067).
- Jena & Rayguru 2026 compare sensitivity-locked policies (https://pmc.ncbi.nlm.nih.gov/articles/PMC13250719/).
- Hasan et al. 2026 study ordinal conformal uncertainty under shift (https://pmc.ncbi.nlm.nih.gov/articles/PMC13512306/).
- MICCAI 2026 RAO-RL addresses risk of underestimating advanced DR (https://papers.miccai.org/miccai-2026/0030-Paper0916.html).
- MICCAI 2026 OMIA restoration study evaluates quality triage (https://papers.miccai.org/miccai-2026-sat/OMIA_001.html).

**No claim that three actions, quality assessment, risk aversion, selective inference or domain shift testing are themselves new.** The exact double-budget decision problem remains a *candidate* gap.

## Aims
Primary H1: At predeclared (review, recapture) operational budgets, quality-and-severity-dependent policy improves the rate of severe DR *not* auto-cleared compared with independently calibrated quality-first, confidence-only and risk-only policies.
Supporting H2: Detailed clarity/artifact/field information improves severity-specific triage utility versus only a scalar quality score.
Supporting H3: Quality- and risk-triggered second-pass inference reduces CPU mean latency without a material deterioration in the primary endpoint.

## Data and endpoint
Source: DeepDRiD regular images, where patient and study source are traceable; optional EyeQ/EyePACS or DDR after checking overlaps and terms. IDRiD is mostly adequately gradable and serves external disease validation, not a mixed-quality benchmark. Messidor-2 has no official grades; any third-party labels must be separately cited and checked.
Use individual eyes/images as input to a **patient-level** split where identifiers permit. For bilateral/multiview images, keep all views/eyes from a patient together. Consider patient-level aggregation as a secondary analysis (if full patient coverage). Avoid treating repeated captures as independent patients.
Primary referable-DR *proxy* = ICDR grade >=2, noting public DR grading labels do not reliably identify diabetic macular edema and independent ocular disease.
Primary endpoint = missed grade>=2 among routine outputs per 1,000 all screened images, **jointly accompanied by** severe (grade>=3) cases deferred to recapture (possible delay), severe cases missed among routine, auto-clear fraction, clinician-review fraction and recapture fraction. All candidate policies must report both missed-clear and recapture-deferred disease. Use predefined hierarchical co-primary safety interpretation or utility sensitivity analysis; never present recapture of severe disease as a successful referral. Analyze DR grade and image quality strata. A recapture action is NOT counted as a successful diagnosis/referral.

## Fixed evaluation design
- Fit DR + quality model using *train* only.
- Select checkpoints, calibration parameters and operating thresholds using held-out train/validation/calibration data only; maintain disjoint patients.
- Final internal test and all external tests are locked until all choices are frozen.
- Each comparator should be independently tuned under the same constraints (review_budget, recapture_budget, minimum auto-clear coverage); infeasible scenarios must be marked infeasible.
- Predeclare e.g. review budgets 20%, 30%, 40% and recapture budgets 10%, 20%, 30% as exploratory scenario grid; NOT clinical recommendations and not known feasible.
- Evaluate missed RDR per 1000 all images and per 1000 cleared, coverage-risk plots, severe-grade errors, classwise QWK, AUROC, AUPRC, ECE/Brier, group-specific counts.
- Use patient-cluster bootstrap 95% CI and paired bootstrap differences; do not overinterpret tiny severe-disease subgroups.
- Report calibration failures under natural external shifts; never imply conformal marginal guarantees externally if the exchangeability condition is not met.
- Keep risk labels strictly held out from triage prediction pipeline. Do not tune on external test labels.

## Important identifiability limit
Public cross-sectional fundus datasets do not specify whether a recapture attempt *would have succeeded*. Therefore, we can evaluate model-suggested recapture rates and deferred label outcomes, **not true clinical recapture benefit**. We must not report saved visits, avoided harms or improved medical safety without prospective evidence. Optional small blinded retina-specialist review of anonymized samples requires institutional approval/waiver as appropriate.

## Failure and go/no-go criteria
1. If quality + severity labels overlap on too few rare high-risk, low-quality examples, **downgrade** recapture optimizing claim; evaluate selective risk only, with limitations.
2. If same-budget policy fails to beat a tuned simple quality-first rule on held-out data, **report negative finding** and change paper thesis to a transparent comparative benchmark, not invent gains.
3. If uncertainty estimates do not transfer externally, show degradation explicitly and do not advertise robustness guarantees.
4. If training CPU use proves prohibitive, freeze lightweight pretrained backbone, train small heads and prioritize proof-quality evaluation over deeper architecture.

## Integrity
Log exact dataset versions, consent/terms, SHA256 manifests, random seeds, train test patient IDs (private), hyperparameters, weights provenance, run times, and script commands. Follow CLAIM 2024 guidance (https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/). Confirm VIT institutional ethics expectations. Do not publish identifiable images or redistribute dataset photos without permission.
