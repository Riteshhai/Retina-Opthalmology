# QSART retrospective evaluation protocol

Version 0.2, 18 October 2026. This extends the supplied protocol without overwriting it. It specifies the intended scientific evaluation separately from the software development pilot. A source record and executable script do not constitute a completed experiment.

## Research question and evidence boundary

Can a policy that uses DR severity probabilities, image quality and uncertainty improve the distribution of automated errors and deferred cases under separate specialist-review and recapture capacity limits? The primary comparison concerns outcomes visible in retrospective labels. It cannot measure treatment benefit, successful recapture or successful specialist diagnosis.

The candidate contribution is a carefully controlled comparison of a three-action policy under two capacity limits. Image quality assessment, quality-aware diagnosis, severity-weighted errors, abstention, external testing, workflow integration and adaptive expert routing each have prior art. Close records include P21, P24, R09–R15 in [the verification matrix](LITERATURE_VERIFICATION.md). Avoid “first”, “clinically safe” and guaranteed workload/diagnostic benefit claims unless the requisite evidence exists.

H1: under predeclared review and recapture budgets, the joint policy reduces missed referable disease among automated negative outputs, without hiding severe cases in recapture. H2: real quality dimensions add information beyond a scalar quality score. H3: selectively applying additional inference reduces measured CPU cost at a prespecified acceptable difference in retrospective endpoints. These are falsifiable hypotheses. Report a negative or inconclusive result when supported.

## Cohort and labels

DeepDRiD regular fundus images are the initial development source. The official public image route and repository CC BY-SA 4.0 license were verified. EyeQ/EyePACS, DDR and IDRiD remain optional follow-up sources under the checks in [the data access audit](DATA_ACCESS_AND_TERMS.md). Do not claim a data availability blocker for DeepDRiD or claim optional datasets were acquired before file verification.

Retain original label fields. A normalized record needs image ID, verified patient/eye/view grouping, original partition, source cohort, DR label provenance, real quality labels, image checksum and transform history. Use an eye-specific grade where the source defines it; do not replace it with the maximum of both eyes. If a grade describes the eye across views, disclose that it is not a per-image lesion annotation. Treat ungradable, missing and inconsistent labels distinctly. Do not infer a grade for an ungradable photo.

Five DR labels follow the source ICDR coding 0–4. For this study, referable-DR proxy is grade >=2 and severe-DR analysis is grade >=3. These operational proxies omit DME and other diseases that may warrant referral. They are not complete clinical referral rules. The supervisor should review their intended interpretation. Use source-supported quality direction/ranges and record the mapping explicitly. Unsupported “illumination” targets must not be invented from blur or overall quality.

## Data separation and pilot handling

Keep official held-out validation/evaluation cohorts intact. Within official training data, create patient-disjoint training, model-selection and calibration subsets with a fixed seed. Retain all views and both eyes from a patient together. Report class and quality counts at the patient, eye and image levels. Verify that grouping semantics come from source documentation; otherwise document unknown identity and use conservative duplicate checks.

A deliberately small, class-enriched subset is allowed for a CPU feasibility pilot. State its selection procedure, acquired and eligible counts, exclusions and missing subgroups. It is not a representative prevalence sample. Small pilot test data may be exercised during implementation, but then becomes development evidence and cannot be reused as the final confirmatory test. Before the full study, freeze new test data or re-establish a documented untouched official test boundary.

Perform exact image-hash and perceptual duplicate checks before fitting. Keep duplicates and known patient groups together. No cropping variant, augmentation or synthetic corruption of a test photograph may enter training. EyeQ derives from EyePACS; treat common origin and any retinal pretraining overlap as potential leakage. Pooling external data into training prevents describing it later as independent external testing.

## Models and calibration

Start with an actual lightweight DR model and a separate or shared quality predictor. Compare backbones only through validation. Log architecture, parameter count, initialization and pretrained weight license/provenance, input resolution, optimizer, class balancing, augmentations, early stopping, checkpoint selection, seeds and hardware. Random initialization and frozen-backbone fitting are valid pilot choices when disclosed. Successful fitting on real photos establishes training execution, not useful diagnostic performance.

Fit temperature scaling on calibration patients after checkpoint selection. For a five-grade vector p, report p(referable)=sum(p2,p3,p4) and p(severe)=sum(p3,p4). Predictive entropy normalized by log(5) is a score in [0,1], not a calibrated probability of error. Report uncalibrated and calibrated Brier score, negative log-likelihood and fixed-bin ECE with sample counts; add reliability plots and sensitivity to binning. Assess calibration again under external shift without silently retuning test-time temperature.

## Actions and allocation

Each image receives exactly one of `AUTO_SCREEN`, `RECAPTURE` or `SPECIALIST_REVIEW`. `AUTO_SCREEN` includes a provisional five-grade model output; it is not synonymous with a negative clearance. Define automated negative as `AUTO_SCREEN` with predicted grade <2. A deferred case remains undiagnosed in this evaluation.

For a fixed batch of N inputs let x(i,a) indicate action a. Enforce one action per image and capacities sum(x(i,review)) <= floor(N B_review) and sum(x(i,recapture)) <= floor(N B_recapture). Record the batch population and capacity rounding. A single-image threshold rule cannot guarantee population workload, especially under shift; describe it separately from batch allocation.

The current software allocation minimizes a proxy sum of action costs. Its grade weights (1,1,3,6,10), quality contribution, uncertainty coefficient and deferral offsets are exploratory model-development assumptions, not measured clinical costs. Document the exact implemented objective with the code/configuration revision in each experiment. No objective may assign verified successful treatment to review or successful improvement to recapture. Evaluate broad cost-weight sensitivity, including stronger penalties for severe cases sent to recapture. Report instability rather than selecting one favorable weighting.

If quality requirements make the capacity constraints infeasible, return and count infeasibility. Do not silently relax quality requirements or convert the overflow to negative outputs. Batch capacities specify operational ceilings, not a recommended clinical service policy. A capacity-limited model output cannot establish adequate access to care.

## Baselines and fair comparisons

Compare plain grading, confidence-only, quality-only, severity-only, single-budget deferral, naive quality-recapture plus uncertainty review, and the proposed joint policy. Preserve the plain classifier as an unselective reference. Independently tune comparator thresholds and any permissible score weights using calibration/validation patients only. Give each comparator the same search budget and usable inputs except for its declared ablation.

Use a predeclared scenario grid, initially review ceilings 0.20/0.30/0.40 and recapture ceilings 0.10/0.20/0.30. These are exploratory operational scenarios and may be infeasible. Report requested ceilings and realized rates separately. Same ceilings do not imply equal realized workloads. Plot risk at matched realized review/recapture rates or report discrepancies and an a priori matching tolerance. Methods that lack a recapture action cannot support an equal-recapture claim at positive recapture use; include a separate zero-recapture scenario and identify that structural limitation.

Do not optimize the policy using ground-truth grade or quality labels of the cases being triaged. Ground-truth labels enter evaluation only. If batch allocation uses all unlabeled test scores jointly, explicitly state this transductive operating assumption and compare all batch policies under it.

## Endpoints and denominators

The principal descriptive endpoint is 1000 times the number of images with true grade >=2 that receive an automated negative, divided by **all** screened images. Accompany it with the corresponding rate among automated negatives and the numerator/denominator counts. Report severe-grade automated negatives separately.

Always report referable and severe cases assigned to recapture, referable/severe cases assigned to specialist review, positive `AUTO_SCREEN` results, total coverage, quality of automated outputs, both workloads and all infeasibilities. Recapture of severe disease is a possible delay, not a successful referral. Do not add deferred cases to true positives under an assumed perfect specialist. Retained-set sensitivity/specificity must be explicitly labeled and must not be presented as all-screened sensitivity/specificity.

For grading report accuracy, macro-F1, class recall and confusion matrices, QWK and binary referable AUROC/AUPRC where both classes exist. For quality report source-appropriate classwise metrics or continuous error, and performance for each real supported defect. Report missing/undefined metrics as unavailable, not zero. Include natural quality-by-grade cell counts and avoid strong inference from sparse cells.

Use paired patient-cluster bootstrap confidence intervals and differences where patient IDs are authoritative. Resample a patient with all its eyes/views. State the number of resamples, random seed and interval construction, and flag degenerate rare-event intervals. A pilot with very few independent patients cannot support precise subgroup estimates. These intervals do not cure selection bias. Full-study primary hypotheses, comparisons and multiplicity handling must be frozen before opening final labels.

## Quality and compute experiments

Analyze natural defects before controlled synthetic degradations. Corruptions receive separate labels and severity levels and remain a robustness supplement. Disease can itself degrade visibility; quality-severity association is not proof that acquisition failure caused model error. Do not claim that enhancement restores real pathology.

For conditional inference compare always-fast, always-expensive, uncertainty-triggered and quality/severity-triggered paths. Select activation thresholds before the test phase and log whether the second pass changes predictions or actions. Record warmup, thread count, batch size, image decoding/preprocessing, p50/p95/mean CPU time, memory, model size and activation fraction on the named hardware. Distinguish model-only time from end-to-end API latency. A valid scientific efficiency claim needs measured accuracy/risk differences at matched workload, not fewer theoretical FLOPs alone.

## Reproducibility and release gate

Every run needs immutable config, source revision, dataset manifest hash, split/group ID hashes, checkpoint digest, calibration artifact, run command, hardware and machine-readable metrics. Keep identifiable records/photos, credentials and restricted artifacts out of Git; retain permitted attribution. Confirm institutional expectations for secondary analysis of the chosen de-identified public datasets and any prospective/reader study separately. A hospital deployment, physician validation or ethics approval must never be implied by local API or Docker success.

Software fixtures, real-image pilot results and final research results must have distinct status labels. CLAIM 2024 [P16] informs reporting. The companion draft is a protocol/manuscript source with unfilled results, and must remain so until reproducible experimental outputs exist.
