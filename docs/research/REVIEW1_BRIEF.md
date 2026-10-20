# QSART Review 1 briefing

Updated 20 October 2026 for the 20–23 October review window. The current deliverable
includes genuine training, locked internal confirmation and a verified local
Docker application. This is a research prototype with no clinical validation.

## Research question

How should provisional retinal grading distribute automated errors and unresolved
cases when specialist review and image recapture have separate capacities? QSART
implements AUTO_SCREEN, RECAPTURE and SPECIALIST_REVIEW. AUTO_SCREEN includes
positive provisional grades; neither deferred action means completed care.

The candidate contribution is an explicit two-budget comparison with separate
accounting for severe cases sent to recapture. Quality-aware grading, abstention,
severity weighting and adaptive computation have substantial prior work. The
43-record primary-linked literature matrix retains all 28 original paper IDs;
it does not establish that QSART is the first such system or publication-ready.

## Completed evidence

| Area | Actual evidence |
| --- | --- |
| Local project | Source, environment, data, weights, reports and scripts in QSART under MAJOR PROJECT CV; original Office assets preserved |
| Development data | 1,184 real images / 296 official-training patients: 848 fitting, 168 selection, 168 calibration |
| New confirmation | 340 images / 85 official-validation patients; excludes all 15 inspected pilot holdout patients; grouping and cross-cohort fingerprint checks passed |
| DR comparison | Frozen ImageNet MobileNetV3-small versus ResNet18 at 224px; selection macro-F1 0.537 versus 0.516; declared rule selects MobileNet |
| Quality learning | Binary overall quality plus actual clarity, field-definition and artifact annotations; classical 11-statistic logistic comparator |
| Calibration | Separate 168-image partition; original and flip-TTA temperatures; artifact and exact source hashes frozen before confirmation |
| Software | 89 tests passed with current real-model integration; Ruff passed |
| Application | FastAPI + five Streamlit pages, real uploads, preprocessing, probabilities, uncertainty, cohort budgets and CSV/JSON exports |
| Docker | Rebuilt CPU image; actual single/batch/TTA, readiness, model card, rejection and dashboard health passed; all 31 runtime Python files match host |

## Reserved internal confirmation

| Measure | Result |
| --- | ---: |
| Five-grade accuracy | 60.00% (204/340) |
| Patient-bootstrap 95% accuracy interval | 52.65–67.06% |
| DR macro-F1 / quadratic weighted kappa | 0.53176 / 0.72650 |
| Referable DR AUROC / AUPRC | 0.92482 / 0.90789 |
| DR ECE, raw → calibrated | 0.08790 → 0.03999 |
| Overall-quality accuracy / macro-F1 | 74.12% / 0.74028 |
| Classical-quality accuracy / macro-F1 | 73.24% / 0.73133 |
| Clarity / field-definition / artifact accuracy | 49.41% / 48.53% / 35.59% |

Defect predictions remain weak; artifact accuracy falls below majority-class
accuracy (47.94%). Grade 4 has only 14 images. Intervals resample 85 patient clusters
1,000 times, retaining all views. Same-domain confirmation does not establish
external transport or prospective utility. Calibration-fit diagnostics are kept
separate from these independent prediction metrics.

The original three-epoch random-init 96px pilot predicted one DR grade on every
holdout image (16.67% accuracy) and all images poor (50% quality accuracy). Those
reports are preserved. Different populations and protocols prevent treating the
two percentages as a paired improvement estimate.

## Policy findings and CPU evidence

Seven policies across all nine declared capacity pairs yielded 27/63 infeasible
combinations. No model, threshold or cost was tuned on confirmation. Compare actual
workloads, not only nominal ceilings. At 40% review / 20% recapture, quality-only,
naive and joint methods each automate 136/340 images. Their missed-referable
automated negatives are 12, 5 and 5; severe cases assigned to recapture are 18, 18
and 0. The joint-minus-quality-only missed-referable difference is −20.59 per 1,000
all inputs (exploratory patient-bootstrap interval −35.37 to −8.82).

The joint and naive automated outcomes are identical at this point. Full-grid
results, unfavorable outcomes, conditional intervals and multiplicity limits are
retained in [current results](CONFIRMATION_V1_RESULTS.md). Recapture and specialist
review remain unresolved recommendations; no patient benefit is inferred.

On the i5-12500H, two CPU threads, 224px, 48 selection images and three repeats,
mean preprocessing/model time in ms/image was 194.72 single, 188.15 all-image TTA,
182.78 uncertainty, 169.51 severity and 182.29 budget. Activation was 0%, 100%,
22.92%, 25% and 25%. The counterintuitive timing order demonstrates workstation
and sequential-order variability; it does not establish a speed advantage.
JPEG decoding, HTTP and triage are excluded. RSS was sampled at batch boundaries
(about 1,015–1,016 MiB), rather than continuously measured.

## Live demonstration

Open http://127.0.0.1:18501/ and http://127.0.0.1:18000/docs on this workstation.
Show the loaded model/version and internal confirmation scope; then upload a
local authorized fundus image, inspect preprocessing, probabilities and provisional
action. Demonstrate a small batch with feasible capacities and exported results.
Explain explicit infeasibility for insufficient mandatory-deferral capacity.
Readiness establishes artifact loading, not screening accuracy.

## Next evidence gates

The now-inspected confirmation must not become an undisclosed tuning set. Next
work includes independent comparator development, repeated seeds, controlled
quality degradation, stronger defect learning and separately acquired external
data with verified access terms and overlap checks. Patient-level referral,
actual specialist/recapture outcomes and prospective utility remain open.

Overall project acceptance: **PARTIALLY COMPLETE**. See [architecture](ARCHITECTURE.md),
[protocol](DEVELOPMENT_V1_PROTOCOL.md), [current results](CONFIRMATION_V1_RESULTS.md),
PROGRESS.md and the original frozen pilot reports. Final author, affiliation,
ethics/waiver, funding and submission declarations require the actual researchers.
