# Development and internal confirmation — 20 October 2026

This report supersedes the pilot as the current model evidence. The original
pilot reports remain unchanged. The two rounds have different cohorts and
training protocols, so their accuracy difference is not a paired improvement
estimate. Neither round provides external or clinical validation.

## Frozen design and provenance

The [development protocol](DEVELOPMENT_V1_PROTOCOL.md) was declared before new
confirmation predictions. All 296 unambiguous official-training patients supplied
1,184 images: 848 fitting / 212 patients, 168 selection / 42 patients and 168
calibration / 42 patients. All known views and both eyes stayed together; connected
texture-fingerprint groups also stayed together. Four ambiguous subjects were
excluded rather than relabeled.

The new confirmation contains 340 images / 85 official-validation patients. It
excludes all 15 inspected pilot holdout patients. SHA and texture-fingerprint checks
found no cross-cohort candidates at the declared distance. This is an internal,
same-domain population; final official test data were not used.

ImageNet MobileNetV3-small and ResNet18 backbones were frozen, including BatchNorm
statistics. New linear DR heads used 224px inputs, original/flipped fitting views,
class-weighted AdamW, seed 2026 and early stopping. MobileNet won the predeclared
selection macro-F1 rule (0.53724 versus 0.51601); this comparison used no confirmation
predictions. The quality MobileNet learned binary overall quality and real mapped
clarity, field-definition and artifact classes. A fixed standardized 11-statistic
logistic quality comparator was fitted using training data only.

Temperatures were fitted on the dedicated calibration group after model selection.
First-pass and flip-TTA temperatures are separate. Calibration-fit diagnostics are
not independent accuracy estimates. The [selection lock](../../reports/development_v1/selection_lock.json)
hashes checkpoints, calibration, manifests, protocol, policy and evaluation code
before confirmation. Its `confirmation_predictions_inspected: false` records the
state at freezing, not the state after evaluation. Source and all public reports
remain auditable; patient identifiers, per-image results and weights remain local.

## Reserved confirmation results

| Measure | Observed value |
| --- | ---: |
| Five-grade accuracy | 204/340 = 60.00% |
| Patient-cluster bootstrap accuracy interval | 52.65%–67.06% |
| Macro-F1 | 0.53176 |
| Quadratic weighted kappa | 0.72650 |
| Referable DR AUROC / AUPRC | 0.92482 / 0.90789 |
| DR ECE, raw → calibrated | 0.08790 → 0.03999 |
| Training-prior constant-grade accuracy | 46.47% |
| Overall-quality accuracy / macro-F1 | 74.12% / 0.74028 |
| Overall-quality poor-class AUROC | 0.81026 |
| Classical quality accuracy / macro-F1 | 73.24% / 0.73133 |
| Classical quality poor-class AUROC | 0.79297 |
| Clarity accuracy / macro-F1 | 49.41% / 0.37789 |
| Field-definition accuracy / macro-F1 | 48.53% / 0.33558 |
| Artifact accuracy / macro-F1 | 35.59% / 0.27557 |

The accuracy interval uses 1,000 patient-cluster resamples with seed 2026. It does
not account for training randomness or domain shift. Grade-specific recall is
72.78%, 52.63%, 41.67%, 48.28% and 78.57% for grades 0–4; grade 4 has only 14
images. No model improvement significance test is inferred from unpaired point
estimates. The artifact head is worse in accuracy than predicting the majority
class for every image (163/340 = 47.94%). Field-definition class 0 has no observed
confirmation images; its fixed-class macro-F1 includes this absent class. These
heads do not establish reliable defect detection. The overall-quality deep versus
classical difference is small and has no declared superiority conclusion.

See [the frozen aggregate report](../../reports/confirmation_v1/evaluation.json),
[grading confusion](../../reports/confirmation_v1/grading_confusion.png),
[quality confusion](../../reports/confirmation_v1/quality_confusion.png) and
[natural quality/severity plot](../../reports/confirmation_v1/severity_quality.png).
The [reliability plot](../../reports/confirmation_v1/dr_reliability.png) uses the
same fixed ten confidence bins for raw and temperature-scaled probabilities.

Supplementary descriptive analysis of the frozen predictions produces 97
[natural-defect strata](../../reports/confirmation_v1/natural_defect_strata.csv),
including each observed source code crossed with DR grade and patient-cluster
accuracy intervals. Poor-quality images have DR accuracy 58.51% (188 images),
versus 61.84% for good-quality images (152 images). Unselected first-pass
referable false-negative rates are 14/74 = 18.92% and 14/70 = 20.00%, respectively.
These marginal results do not support a general claim that poor quality always
increases false negatives. Quality, severity and acquisition are confounded;
the strata are associations, not controlled damage experiments. Entropy detects
grading errors with AUROC 0.76521, but is not an individual error probability.
This supplementary analysis was added after confirmation and made no model or
policy changes; small strata and multiplicity limits are explicitly flagged.

## Fixed policy grid and unresolved outcomes

Seven policies were evaluated at all nine predeclared review/recapture budget
pairs; 27 of 63 combinations were infeasible. No thresholds, proxy costs or budget
choices were tuned on confirmation. The full table and
[workload plot](../../reports/confirmation_v1/workload_outcomes.png) retain every
feasible operating point. Equal ceilings frequently produce unequal realized
workloads; paired comparisons are made only where actual action rates match.

For illustration, at 40% review / 20% recapture all three methods below realize
those exact rates and retain 136/340 images for provisional automated grading:

| Policy | Missed referable automated negatives | Severe cases sent to recapture |
| --- | ---: | ---: |
| Quality-only | 12 | 18 |
| Naive quality + uncertainty | 5 | 18 |
| Dual-budget joint allocation | 5 | 0 |

Joint minus quality-only missed-referable rate is −20.59 per 1,000 all inputs
(patient-bootstrap interval −35.37 to −8.82); severe-recapture difference is
−52.94 per 1,000 (−85.29 to −23.53). Against naive, automated outcomes are
identical at this point, while severe recapture is lower. These are exploratory
comparisons across a grid, without multiplicity correction, conditional on the
original cohort assignments. They do not demonstrate universal dominance or
successful specialist/recapture outcomes. Both deferred actions remain unresolved.

## Reproduction and remaining evidence

Run `scripts/prepare_development_data.py`, `scripts/train_transfer.py` for both DR
candidates and the multihead quality model, and `scripts/train_classical_quality.py`.
Then run `scripts/finalize_development.py`, `scripts/evaluate_confirmation.py` and
`scripts/package_confirmation.py`. The frozen lock/evaluation deliberately refuses
overwriting or silently changing an evaluated experiment; use a newly declared
experiment directory for a future study. Commands and configuration details are
in [transfer status](../TRANSFER_STATUS.md) and the protocol.

CPU strategy measurements use a 48-image deterministic prefix of selection data,
separate from confirmation. They cannot validate accuracy on a new population.
Further work includes independent comparator tuning on development data,
controlled degradation studies, repeated training seeds, external-domain data,
patient-level referral logic and prospective workflow evidence. Confirmation is
now consumed evidence and must not become an unacknowledged new tuning set.
