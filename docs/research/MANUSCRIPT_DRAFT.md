# Evaluating quality and severity aware retinal triage under separate review and recapture budgets

**Protocol and engineering-pilot manuscript draft, 18 October 2026.** This is an IEEE-compatible section and citation source, not a completed confirmatory study. Citation keys refer to `references.bib`; final venue formatting and complete author/affiliation metadata remain to be applied. Preliminary results below come from the saved real-image pilot artifacts and establish pipeline execution, not a clinically useful model.

## Abstract

Retrospective evaluation of diabetic retinopathy systems requires explicit treatment of image quality, uncertain predictions and outcomes deferred to human care. We implement a three-action research policy under separate specialist-review and image-recapture capacities, with lightweight DR and quality models and held-out temperature fitting. A genuine DeepDRiD engineering pilot used 160 fitting images and separate selection, calibration and 60-image/15-patient holdout partitions. After three 96px CPU training epochs from random initialization, the frozen DR model predicted grade 3 for every holdout image, with accuracy 16.67%, macro-F1 0.05714 and QWK 0. The quality model predicted poor for every holdout image, giving 50% accuracy. Seven policies across nine capacity scenarios produced 51 infeasible combinations; feasible selective cases had no automated coverage. Thus the pilot establishes executable training, inference and reporting, while failing to demonstrate useful screening performance or policy benefit. Cross-sectional labels do not reveal successful recapture or specialist diagnosis. Confirmatory comparison, external testing and clinical validation remain outstanding.

**Index terms:** diabetic retinopathy, fundus image quality, selective prediction, calibration, resource allocation.

## I Introduction

Image classification metrics describe one component of retinal screening. The quality of acquisition and the operational handling of uncertain or ungradable images also influence which patients remain unresolved. Existing quality reviews identify inconsistent annotation and evaluation practices [P01]. Field studies illustrate why including ungradable inputs and specifying camera/reader conditions matter [P07, P17]. In this project a grade >=2 is an operational referable-DR proxy, while DME and other causes of referral remain outside the five-grade output [P08, P15].

This study asks a narrower comparative question: how do three-action policies distribute automated errors and deferred disease when specialist review and image recapture each have a separate capacity limit? We treat deferred outcomes as unknown. We will compare policies on a locked retrospective cohort, report quality and severity strata, and measure actual CPU execution. The intended contribution is the evaluation and decision formulation, conditional on results; a new combination of familiar model components is insufficient evidence of novelty.

## II Related work

EyeQ and FundusQ-Net supply established quality grading approaches [R02, P02]. DeepDRiD directly supports joint investigation of DR grades and natural quality labels [R01]. Quality-aware diagnostic representation learning predates this project [R09], and multitask retinal systems already combine quality and disease outputs [P22]. QGDR examines quality-severity coupling and quality-conditioned expert routing [R10]. Therefore QSART cannot claim the first quality-aware classifier, quality-severity analysis or adaptive quality-dependent computation.

GDRNet, CauDR and disentangled representation methods address cross-domain DR generalization [P03, P04, P09]. RETFound supplies a retinal representation-learning comparator [P06], and FOCUS examines quality and generalization across model families [P25]. These findings motivate external testing without establishing that an unseen population will share calibration or workload properties with development data.

Selective classification and deep ensembles are established methods [R07, R06]. DR-specific entropy abstention [P21], Bayesian uncertainty [P18] and ordinal conformal prediction [P24] further narrow possible claims. Temperature scaling is a suitable transparent baseline [R05]. The Feng et al. publisher abstract describes quality-aware ordinal grading, calibration and uncertainty-guided referral [R12]; full-method comparison remains necessary before a novelty claim. NaIA-RD demonstrates integrated screening workflow [R11], and Jena and Rayguru compare screening decision strategies [R13]. Risk-averse ordinal grading and synthetic restoration/triage studies provide further close comparisons [R14, R15].

The remaining candidate question is whether an explicitly reported two-budget policy offers useful held-out trade-offs compared with tuned simple alternatives, including the unresolved severe cases assigned to recapture. This audit does not establish an exhaustive absence of equivalent prior work.

## III Materials and study design

The development source is regular-fundus DeepDRiD at revision `56d8af71319803150d0e7b90533e0918c7bf85ee` [R01]. The authors publicly serve images and annotations under repository CC BY-SA 4.0. The acquired pilot contains 260 JPEGs from 65 patients with all four known images each: 160 fitting images/40 patients, 20 selection images/5 patients and 20 calibration images/5 patients from official training, plus 60 holdout images/15 patients from official validation. Sampling was enriched by class and excludes four source subjects affected by ambiguous annotation rows. It does not estimate screening prevalence. Manifest hash, file verification and source-conflict quarantine details are documented in [data status](../DATA_STATUS.md).

All verified views and both eyes of a patient remain in one partition. Official held-out splits remain separate. Development training is subdivided into fitting, model selection and calibration groups. Any set inspected during iterative debugging is development evidence; final inference requires untouched test labels and frozen choices. An optional external cohort will be named only after access, label harmonization and overlap checks. DDR is a candidate source [R03]; IDRiD acquisition and current archive terms still need confirmation [R04]. EyeQ and EyePACS share source photographs and cannot form a naive independent train/test pair [R02].

Five-class disease grades follow the source's eye-level definition; they are not independently assigned lesion labels for each view. Predicted referable risk sums grades 2–4. Overall-quality labels are 0, not good enough, and 1, good enough. The holdout has 30 images in each quality class and grade counts 16/8/20/10/6 for grades 0–4. Source clarity, field and artifact codes remain in the manifest but are not learned by this first binary-quality checkpoint. Code mapping and separate defect-head experiments remain future work.

## IV Methods

### A Predictors and calibration

The implemented pilot independently trains a five-grade MobileNetV3-small DR model and a binary overall-quality MobileNetV3-small model [R08]. Both use random initialization, 96px shared crop/letterbox preprocessing, batch size 8, two CPU threads, seed 42 and three epochs with class-balanced cross-entropy. Each run completed 60 optimizer steps. Validation loss selected DR epoch 1 (20 steps) and quality epoch 2 (40 steps); later training steps do not replace the selected checkpoints. These settings were not selected on holdout performance. Candidate-backbone and initialization comparisons remain incomplete.

Temperature scaling was fitted on the 20 calibration images [R05], including separate temperatures for original and horizontal-flip mean-logit paths. Normalized entropy is an uncertainty score, not a calibrated error probability. The selected DR temperature was 0.159717 and quality reached the configured maximum of 20. Calibration-fit and held-out diagnostics are reported separately below. No calibration or policy threshold was refitted from holdout outputs.

### B Two-budget allocation

For N images the decision variable x(i,a) assigns one action a to each image. The actions are automated provisional grading A, recapture C and specialist review R. The constraints are sum_a x(i,a)=1, sum_i x(i,R)<=floor(N B_R) and sum_i x(i,C)<=floor(N B_C). Action costs are estimated from model outputs and explicit exploratory severity/deferral weights. These weights are assumptions, not measured clinical utilities. The current software solves a cohort assignment problem; a separate pointwise rule has no population budget guarantee.

An image receiving `AUTO_SCREEN` retains its five-grade prediction. Automated negative means `AUTO_SCREEN` and predicted grade below the referable threshold. Recapture and review remain deferred outcomes. A lack of capacity is reported as infeasibility when mandatory quality deferrals cannot be accommodated. No perfect specialist or successful recapture is assumed. The implementation revision and exact objective will be archived with each run.

### C Comparators and matched workload

The pilot executes plain classification, confidence-only, quality-only, severity-only, single-budget review, naive quality recapture plus uncertainty review and joint allocation over review ceilings 0.20/0.40/0.60 and recapture ceilings 0.10/0.20/0.40. This engineering grid and its fixed proxy costs were not retuned from holdout predictions. It is a separate exploratory grid from the companion protocol's suggested full-study grid. Independent comparator hyperparameter optimization and a confirmatory matched-rate study remain future work. Equal ceilings need not yield equal realized use. A comparator without recapture needs a distinct zero-recapture analysis for an equal-action-use comparison.

### D Natural quality and CPU analysis

Cross-tabulate natural quality labels, DR grade, errors and actions. Synthetic blur, illumination changes and other controlled corruptions are secondary stress tests, with separate labels and no implied clinical recapture outcome. Conditional inference compares always-fast and always-expensive paths with predeclared trigger rules. Measure end-to-end and model-only CPU latency, activation fraction and memory on the named host. Additional computation is useful only if the measured decision trade-off supports it.

## V Evaluation and statistical analysis

The primary descriptive rate is missed true grade >=2 among automated negatives per 1,000 **all** inputs. Report the same numerator among automated negatives, severe-grade misses, referable/severe cases deferred to recapture, cases assigned to review, automated positive outputs and coverage. Retained-set sensitivity/specificity receive explicit denominators; deferred cases are not true positives by assumption.

DR grading metrics include macro-F1, per-grade recall, QWK, confusion matrix and referable AUROC/AUPRC where defined. Quality metrics follow label scale and include subgroup counts. Use paired patient-cluster bootstrap intervals with all views/eyes resampled together. Sparse and degenerate subgroups are flagged. Report requested budgets, realized action rates, violations and infeasibilities, and show the full prespecified trade-off grid. No thresholds or cost choices are selected on final test labels. Follow the companion protocol and CLAIM reporting guidance [P16].

## VI Results

### A Frozen real-image pilot

The genuine holdout run is `EXP01_EXP04_EXP05_EXP08_pilot`, saved in [evaluation.json](../../reports/pilot/evaluation.json). Its 60 images belong to 15 patients and were separated from fitting, checkpoint selection and temperature fitting. The manifest SHA256 is `e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0`. DR, quality and calibration hashes are respectively `a9ec1aa68c48e37b3cac6131c05e74838d9ecd26fe13e5fa252d2bfbca14c701`, `dcdb35ce2a8dc7951d4397b8020a61b13a4ca68adccf3edb1f7ac447fc0adf07` and `05a0277cb716ba7b83de1aba99d15a92175f7045656c92274d9f126b352eedfb`. No model or policy retuning followed evaluation.

| Held-out measure | Measured value | Limitation |
| --- | ---: | --- |
| DR accuracy | 10/60, 16.67% | All predictions are grade 3 |
| Macro-F1 | 0.05714 | Five-grade discrimination failed |
| QWK | 0 | No agreement beyond this degenerate grade output |
| Referable AUROC / AUPRC | 0.44213 / 0.57690 | Enriched cohort with 36/60 referable images |
| Accuracy 95% interval | 3.33%–33.33% | Percentile bootstrap, 1,000 resamples of 15 patient clusters, seed 2026 |
| Overall-quality accuracy | 30/60, 50% | All predictions poor; equals the class baseline |

The pilot demonstrates actual optimization and serialization but fails to show useful predictive ability. Natural quality/severity cell counts and error heatmaps are generated; the single-class disease prediction prevents a substantive quality-failure mechanism claim.

### B Calibration diagnostics

On the 20 calibration-fit images DR NLL fell from 1.605348 to 1.595836, while 15-bin ECE rose from 0.005203 to 0.032086. Quality NLL fell from 0.701899 to 0.693547 and ECE fell from 0.119890 to 0.100995. These are optimization-sample diagnostics from five patients, not independent calibration performance. On the holdout, DR ECE was 0.065419 calibrated versus 0.038536 raw and multiclass Brier was 0.807680 versus 0.800328. Calibration therefore did not yield a general reliability improvement in this pilot. Fitting NLL does not guarantee an improvement in ECE or discrimination.

### C Allocation and ablations

The [budget comparison table](../../reports/pilot/budget_comparisons.csv) contains 63 rows: seven policies across nine pairs. Fifty-one rows report infeasibility rather than silently relaxing quality restrictions. The plain classifier is evaluated at all nine ceiling pairs with zero deferral. Its reported zero automated missed-referable cases occurs because every grade is positive; specificity is zero. That rate is not evidence of successful screening.

Only quality-only, naive and joint selective allocation are feasible at review 0.60/recapture 0.40; each defers all 60 cases and has zero automated coverage. Quality-only and naive allocation each place 4 referable cases and 0 severe cases in recapture. Joint allocation places 12 referable cases, including 7 severe cases, in recapture. This unfavorable allocation is retained as observed; recapture success is unknown. The pilot cannot support joint-policy superiority, and zero automated error at zero automated coverage provides no such evidence. The [ablation table](../../reports/pilot/ablations.csv) likewise records infeasible variants; the feasible no-quality variant remains limited by the same all-grade-3 model. It is not a completed independently optimized scientific ablation study.

### D CPU observations

Actual CPU benchmarks in `artifacts/pilot_cpu_benchmark.json` use 20 selection-validation images, three repeats and one excluded warmup image on Intel Core i5-12500H, Windows build 26200, PyTorch 2.14.1+cpu and two torch threads. Requests are grouped in chunks of eight; individual model forwards have batch size one.

| Inference strategy | Mean ms/image | P95 ms/image | Extra-pass fraction |
| --- | ---: | ---: | ---: |
| Single | 230.94 | 328.39 | 0 |
| Horizontal-flip mean logits for all | 242.84 | 345.99 | 1.00 |
| Uncertainty trigger with cap | 289.73 | 458.95 | 0.25 |
| Severity trigger with cap | 280.75 | 490.63 | 0.25 |
| Ranked population cap | 231.05 | 342.37 | 0.25 |

Timing includes shared image preprocessing and both DR/quality model passes, excluding JPEG decoding, HTTP transport and triage. RSS sampled at chunk boundaries was approximately 636–638 MiB and is not a continuously measured peak. Workstation-load variation and sequential measurement order prevent a reliable efficiency comparison; fewer second passes did not consistently mean lower observed time. No clinical or diagnostic advantage follows.

Full-cohort training, useful model discrimination, independent comparator tuning, matched-rate confirmation, real defect-head learning and external testing remain outstanding. The initial Docker image has built; latest-source container rebuild/runtime checks and final software/real-artifact integration tests were pending at this manuscript update. Deployment remains unverified until their actual results are recorded.

## VII Discussion

The observed single-class predictions are a failed predictive result within a successful engineering execution. The small class-enriched cohort, random initialization, 96px input and three-epoch schedule limit the inference, and this experiment cannot distinguish their causal contributions to failure. The holdout must now remain an inspected pilot result rather than a target for repeated optimization. Further development uses fitting/selection data and a separate frozen final test. A development pilot cannot justify publication-ready diagnostic accuracy, improved safety or superior resource allocation. Quality can correlate with pathology as well as acquisition conditions, so association is not a causal estimate of image damage.

Recapture recommendations and review allocations are observable software decisions. Their success, delays, staffing consequences and patient outcomes require additional evidence. A triage score must not hide unresolved severe disease behind improved retained-set accuracy. Patient-level referral logic, DME and other retinal disease, reader disagreement, external transportability and prospective workflow validation remain material limitations.

## VIII Reproducibility and ethics

Release code, auditable configurations, dataset source/license notices, split hashes and scripts for rebuilding the permitted local artifacts. Keep photos, identifiers and restricted data out of Git. The training checkpoint code-version field records repository HEAD before the newly implemented source was committed; that SHA alone is not an exact training-source snapshot. Preserve the delivered source/configurations, package versions and histories with the unchanged artifact checksums. The evaluation report separately records an implemented-source digest. Weight provenance and model-card limitations accompany any released checkpoint. Record institutional ethics expectations and any approval/waiver status without inventing a determination. Local Docker/API execution is an engineering result and does not establish clinical authorization. Author contribution, funding, conflicts of interest and any required AI-assistance disclosure remain for the actual authors to complete.

## IX Conclusion

QSART executes real-image training, calibrated CPU inference and explicit three-action cohort allocation. The small pilot's degenerate predictions and unfavorable unresolved-case allocation do not establish screening utility or comparative benefit. A useful trained model and a separately frozen, independently tuned evaluation remain necessary before a scientific contribution can be claimed.

## References

Use the verified citation keys in `references.bib` with IEEEtran or the selected venue's bibliography style. The companion verification matrix distinguishes journal articles, proceedings, author preprints, missing publisher identifiers and access limitations. All 28 supplied records are retained alongside the added primary sources. Final venue formatting requires numerical citations generated from the bibliography.
