# QSART Review 1 briefing

Prepared 18 October 2026 for the 20–23 October 2026 review window. A genuine real-image engineering pilot has completed training, calibration and frozen-model holdout evaluation. Its predictions are degenerate and do not establish useful screening performance. Confirmatory scientific results and clinical validation remain pending.

## Problem and objective

QSART studies how a retinal screening prototype should handle uncertain or low-quality images when specialist review and repeat-image capacity are separately limited. The three software actions are provisional `AUTO_SCREEN`, `RECAPTURE` and `SPECIALIST_REVIEW`. An automated grade is a research output; a deferred case remains unresolved.

The principal hypothesis is fewer missed referable cases among automated negatives at matched review and recapture workloads, while reporting severe cases assigned to recapture as a separate possible-delay outcome. Supporting questions concern natural quality defects by disease grade and measured conditional CPU inference. These are research hypotheses, not achieved clinical outcomes.

## Demonstrable progress

| Area | Verified evidence | Interpretation |
| --- | --- | --- |
| Existing research preparation | Two literature DOCX files, 16-slide PPTX, 28-record paper library and starter code preserved | Reusable proposal assets, not trained-model evidence |
| Primary-source audit | All 28 original IDs retained, 43 total records with primary URLs and DOI/status limits; bibliography source generated | More than 20 primary citation records independently retrieved; individual experimental replications are outside this audit |
| Dataset acquisition | 260 actual DeepDRiD JPEGs from 65 patients; source revision and repository CC BY-SA 4.0 retained | Class-enriched feasibility subset, not representative screening prevalence |
| Patient-disjoint partitions | 160 training images/40 patients; 20 selection images/5 patients; 20 calibration images/5 patients; 60 official-validation holdout images/15 patients | All four known views per patient stay together; final official test untouched |
| Software behavior | 78 tests passed with real-image integration enabled; Ruff passed; actual API response rendered across the five-page Streamlit interface | Generated fixtures test software; the separate real-image results below test the pilot artifacts |
| Real training | DR and overall-quality MobileNetV3-small models, random initialization, 96px, three epochs, batch 8, two CPU threads, seed 42 | 60 optimizer updates per run; validation loss selected DR epoch 1 and quality epoch 2 |
| Calibration | Temperatures fitted on 20 separate calibration images; checkpoint/manifest hashes recorded | DR calibration-fit NLL improved but ECE worsened; these are fitting diagnostics |
| CPU measurement | Real artifact-backed validation inference benchmark completed on Intel i5-12500H | No reliable speed advantage or clinical benefit demonstrated |
| Docker | Latest CPU image built; API and dashboard healthy; readiness, real-image single/batch/TTA and dashboard HTTP checks passed on local ports 18000/18501 | Runtime verification establishes engineering execution, not useful screening performance |

The API/dashboard, checkpoint validation, exact cohort budget allocation and evaluation scripts are implemented. Readiness means required artifacts loaded successfully; it does not certify useful predictions. A single-image routing decision does not guarantee population action budgets. [The architecture source](ARCHITECTURE.md) shows the actual data, model and request flow.

## Actual preliminary findings

The frozen held-out evaluation in [evaluation.json](../../reports/pilot/evaluation.json) contains 60 images from 15 patients. The manifest SHA256 is `e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0`. No model or policy thresholds were retuned using holdout predictions.

| Held-out result | Measured value | Meaning |
| --- | ---: | --- |
| Five-grade accuracy | 10/60 = 16.67% | Every image was predicted as grade 3 |
| Macro-F1 / QWK | 0.05714 / 0 | No useful five-grade discrimination demonstrated |
| Accuracy patient-bootstrap interval | 3.33%–33.33% | 1,000 resamples of 15 patient clusters; small enriched cohort |
| Referable AUROC | 0.44213 | Does not demonstrate useful ranking on this cohort |
| Overall-quality accuracy | 30/60 = 50% | Every image predicted poor; equals the holdout's class baseline |

The unselective classifier reports zero missed referable cases because **every output is positive**. Its specificity is 0. This is not a successful screening result. For the 63 method/scenario combinations (seven policies, nine budget pairs), 51 were infeasible. Three selective policies were feasible only at 60% review/40% recapture and deferred every image, giving zero automated coverage. At that pair the joint policy sent 12 referable cases, including 7 severe cases, to recapture; quality-only and naive policies each sent 4 referable and 0 severe cases to recapture. These observed unfavorable allocations are retained. None demonstrates superiority or successful downstream care. Ablation and natural quality/severity tables are generated, with the same model and sample limitations.

On 20 model-selection images with three repeats, mean preprocessing-plus-model time was 230.94 ms/image for one pass, 242.84 ms for a second pass on all images, and 231.05 ms for the ranked 25% extra-pass budget. Uncertainty and severity triggers were slower (289.73 and 280.75 ms). Timings exclude JPEG decoding, HTTP transport and triage. Memory was sampled at batch boundaries, approximately 636–638 MiB RSS. Live workstation variability and the small sequential benchmark prevent a reliable efficiency claim. See [model status](../MODEL_STATUS.md) for the full table and calibration diagnostics.

## What can be claimed about novelty

Quality assessment, quality-aware grading, abstention, severity weighting and clinical screening integration precede QSART. The audit identifies CVPR quality-aware diagnosis, NaIA-RD, QGDR, uncertainty-aware ordinal referral, RAO-RL and other close prior work. The candidate contribution is the **specified two-budget decision comparison and its unresolved-case accounting**. Do not claim the first three-action system or the first use of image quality or recapture.

## Work required after the pilot

Improve model learning using training and selection partitions, larger permitted cohorts and an explicit backbone/resolution/initialization study. The already opened pilot holdout must not become a repeated tuning target. Independently tune each comparator before a new confirmatory test, preserve official test data, and report risk/workload curves, actual action rates, severe recapture cases, infeasibility and paired uncertainty. External testing requires a separately acquired and overlap-audited cohort. Real defect heads remain untrained in the current binary-quality pilot.

For the review, demonstrate the actual preprocessing, learned checkpoints, calibrated inference, API behavior and generated reports, including the failed predictive behavior. This establishes end-to-end engineering execution. It does not establish a useful screening model. The training checkpoint's code-version field names the repository HEAD before the new source was committed; that field alone is not an exact training-source snapshot. Preserve the delivered source, configuration, environment and histories with the artifact hashes.

## Open evidence gates

No confirmatory improvement, external robustness, successful clinical recapture, specialist diagnosis, prospective utility or deployment approval is established. Optional EyePACS photographs require official access terms. IDRiD primary download/archive terms were not retrieved in this audit; DDR is available through a larger author-provided split archive. These optional access limits do not block the initial DeepDRiD pilot.

Use `PROTOCOL_V0_2.md` for the detailed study design and `MANUSCRIPT_DRAFT.md` for the paper source. Final local test/container evidence is recorded in PROGRESS.md and reports/deployment.json. Do not reinterpret the pilot's measured failures as research benefit.
