# Base-paper study and QSART implementation boundary

Studied on 2026-10-20 against MASTER_SPEC v1.0, sections 5 and 12. This is a
technical study note, not a reproduction claim or a manuscript result section.

## Verified identity and reading scope

Hasan, U.; Nayeem, M. A.; Alghamdi, T. G. (2026), *Uncertainty-Aware AI-Assisted
Diabetic Retinopathy Grading from Fundus Images with Ordinal Conformal
Prediction*, Diagnostics **16**(16), 2645, published 19 August 2026.
[DOI](https://doi.org/10.3390/diagnostics16162645),
[complete primary article on PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13512306/),
[PubMed identity](https://pubmed.ncbi.nlm.nih.gov/42651046/).

The entire article was read: introduction, all related-work subsections,
methods 3.1–3.9 including Algorithm 1 and Proposition 1, results 4.1–4.11,
tables 1–4 and figure captions 1–9, discussion, conclusions, availability and
limitations statements. Publisher access initially failed and PMC briefly
returned a browser challenge; the indexed PMC primary full text subsequently
opened successfully. This note uses original paraphrases and mathematical
description. The article identifies its license as CC BY.

The linked [authors' repository](https://github.com/umarbhasan/conformal-diabetic-retinopathy)
was also inspected at revision `3a7d18d799262098d8166519159d538960bbf118`.
Its `src.py` SHA256 is
`4307702942846381ae2e2e2d72a7baa5d4fd20a604287ec971a43a8ddb26f4fe`.
The downloaded source and GPL-3.0 license stay in ignored local
`artifacts/research/`; author code is not vendored, imported or executed.
QSART's implementation independently expresses the published mathematical
construction. Reading the source does not verify its archived numerical outputs.

## Research question and design

The paper asks whether uncertainty sets calibrated internally remain reliable
externally and whether ordinal intervals improve severe-grade inclusion and
structural coherence. It investigates a post-hoc uncertainty layer rather than
claiming a new grading backbone. Coverage is studied at 90% and 95% targets.
This is a retrospective public-dataset analysis, with no new prospective
screening, recapture or specialist decision outcomes.

| Component | Reported procedure and interpretation |
|---|---|
| Source | APTOS 2019: 3,662 labeled images, repackaged as 2,930 training / 366 conformal calibration / 366 internal test. These are partitions of labeled data, not accessible competition hidden-test labels. |
| Source imbalance | Training counts for grades 0–4: 1,434 / 300 / 808 / 154 / 234. Severe grade 3 is about 5%. |
| External IDRiD | Reported grading cohort 455 images, counts 129 / 22 / 156 / 84 / 64. Severe grade 3 is about 18%. The cohort must be matched to an exact manifest before reproducing; the article's reported count is not a substitute for auditing the original release and exclusions. |
| External Messidor-2 | Secondary independent ViT replication, retaining 1,744 adjudicated-gradable images; counts 1,017 / 270 / 347 / 75 / 35. |
| Architectures | ImageNet-initialized EfficientNet-B0, ResNet-50, ViT-Base; five-way linear classifier and softmax probabilities. |
| Preprocessing | RGB, direct resize to 384×384, ImageNet mean/standard deviation; deterministic evaluation. No fundus-border crop is specified in this protocol. |
| Training augmentation | Horizontal and vertical flips, ±15° rotation, color jitter. |
| Optimization | AdamW, learning rate 3e-4, weight decay 1e-4, cosine schedule, at most 15 epochs; weighted cross-entropy with w_c=N/(5 n_c). |
| Checkpoint selection | A stratified 10% subset of the training partition, patience 4; public code monitors accuracy. This is separate from the 366-image conformal calibration partition. |
| Reported compute | Paper: batch 8, mixed precision, PyTorch 2.10.0+cu128 and timm 1.0.26 on Kaggle. Accelerator identity/peak memory were not retained. The public source CFG instead sets batch 16. This discrepancy must be recorded, not silently resolved as an exact reproduction. |
| Conformal calibration | Held-out APTOS only; no external samples for model fitting, early stopping, tuning or conformal calibration. Probability temperature fitting is not specified; "calibration" here primarily means score-threshold calibration. |

Dataset links in the paper point to repackaged Kaggle datasets. These do not
automatically establish legal access, patient grouping, image completeness,
grading-map compatibility or a match to the original competition release.

## Conventional and ordinal scores

For a fitted classifier p(x), define a label score s(x,y). Fit the threshold on
independent calibration true-label scores. With n independent calibration units,
use **k=ceil((n+1)(1-alpha))**, and the kth smallest score when k≤n; otherwise
use positive infinity. Include candidate y when s(x,y)≤q. Inclusive ties are
conservative. The standard claim is marginal coverage for exchangeable
calibration/test units, conditional on an independently fitted scoring rule.
It does not provide disease-grade-conditional coverage or validity under shift.

The paper compares:

- **LAC:** s(x,y)=1-p_y(x).
- **APS:** sort probabilities descending, then sum mass before y plus a uniform
  random fraction of p_y.
- **RAPS:** APS plus lambda·max(r-k_reg,0), where r is the label's sorted rank.
  The paper tunes k_reg in {1,2,3} and lambda in
  {0,.001,.01,.05,.1,.2,.5,1} on a 70/30 calibration fitting/tuning division,
  selecting small sets subject to the reported coverage tolerance.
- **Ordinal greedy:** start with the modal grade, repeatedly add the
  higher-probability immediately adjacent grade, and stop once the candidate
  true grade is reached. Its score is preceding full mass plus
  (1-u)·p_y, with u uniform on [0,1]. Invert this same nested ordering at q.
  Sharing the image's u across candidate labels preserves a nested prefix.

Each ordinal prefix is an adjacent interval. The article and code conservatively
retain the modal grade if randomized inversion would otherwise be empty.
Adding that fallback cannot decrease coverage, but it changes exact randomized
score-event equivalence for that special empty case. The article's proposition
should therefore be interpreted with this conservative enlargement explicit.

There is also an implementation detail worth testing independently: public
`conformal_quantile` calls `np.quantile(scores, k/n, method="higher")`.
NumPy indexes quantiles on n-1 intervals, so this can select a higher order
statistic than the specified kth score. QSART deliberately uses the exact kth
order statistic, as requested by the finite-sample construction.

## Outcomes, external failures and limits

The paper reports empirical marginal coverage, average set size, contiguity and
grade-specific coverage, with grade 3 emphasized. It also reports classifier
ECE using 15 bins and NLL, descriptive MMD/t-SNE domain comparisons, and set
construction overhead. The two-way bootstrap resamples **images and training
seeds**, with 5,000 resamples; it is not described as a patient-cluster bootstrap.
An independent five-seed ViT analysis uses exact one-sided paired Wilcoxon tests.

Internal coverage is approximately nominal, while external coverage deteriorates.
For example, at the 95% target on ResNet-50/IDRiD, grade-3 coverage is reported as
0.750 with APS and 0.945 with ordinal sets, alongside mean set size increasing
from 2.890 to 3.031. This is the paper's result, **not a QSART measurement**.
Ordinal construction gives contiguous intervals by design, with larger sets;
it does not generally dominate on every grade. Grade 4, at the scale boundary,
does not obtain the same consistent benefit. In the Messidor-2 ViT replication,
the ordinal method also remains substantially under nominal coverage.

The findings support testing an uncertainty representation; they do not establish
clinical reliability. Important limits are small severe-grade cohorts, absent
standardized quality/demographic annotations, unreported patient-level split
control, lack of prospective clinician comparison, and no valid same-model
controlled blur/illumination experiment. External MMD/t-SNE is descriptive and
does not identify the cause of failure. Reported Kaggle timing cannot establish
CPU laptop latency. The full article explicitly frames upper-endpoint referral
as a future workflow interpretation requiring evaluation.

## What QSART implements now

`qsart/uncertainty/conformal.py` provides a conventional **LAC** baseline and
**ordinal_greedy**, a **deterministic paper-inspired adaptation**. The latter
uses full cumulative mass (the paper's u=0 setting), lowest-grade modal ties,
and higher-grade adjacent ties, matching the public code's adjacency convention.
It does not implement randomized APS/RAPS or reproduce the paper's complete
APTOS/backbone/five-seed experiment. It must not be named an exact reproduction.

The module stores method, alpha, exact rank, calibration-unit count, threshold,
tie/fallback rules, scope and caller-provided provenance in deterministic safe
JSON. A null threshold encodes positive infinity. It rejects incompatible
class counts, invalid probabilities/labels, invalid artifacts, nonstandard
NaN/Infinity and duplicate JSON keys. Threshold inclusion uses ≤. Both raw
empty sets and a documented conservative modal fallback are supported; default
fallback is enabled. Returned grade lists are ascending.

Optional **patient-max calibration** uses the largest true-label score over each
patient's supplied views as one calibration unit. Under independent,
exchangeable patient clusters and the same view-sampling protocol, this targets
simultaneous coverage of all views in the next patient. It is not a guarantee
for a selectively routed image, an arbitrary shifted population or each grade.
The number of independent units is the number of patients, not the image count.
Artifacts do not retain patient identifiers.

QSART's API-facing sidecar must bind trained DR/quality checkpoint hashes,
probability-transform/calibration hashes, the development manifest and exact
temperature-fit/conformal split fingerprints, and first-pass versus all-flip
probability paths. Temperature fitting and conformal score fitting must use
patient-disjoint partitions. Fitting a temperature on all 168 v1 calibration
images and then reusing them as independent conformal calibration would not
meet that condition. The old v1 model outputs and evidence remain preserved;
new sidecar calibration is a separate artifact and protocol.

First-pass and always-flipped probabilities need separately fitted thresholds.
Choosing an extra pass conditionally can change the selected distribution;
separately calibrated paths do not establish conditional-routing coverage.
Report that routing analysis empirically and make unavailable/mismatched
uncertainty explicit. A broad set or low upper endpoint is not itself a
validated clinical clearance rule. Budget overload must not force uncertain,
referable or ungradable cases into automated non-referable outcomes.

## Immediate experiments and evidence boundary

1. Freeze temperature/conformal patient partitions, method/alpha and exact
   artifact bindings before fitting. Keep the consumed v1 confirmation cohort
   labeled as exploratory reuse when evaluating a new uncertainty layer.
2. Compare LAC and deterministic ordinal coverage at alpha .10/.05; show all
   five grade denominators, patient-all-views inclusion, empty counts, set-size
   distributions and contiguity. Small patient counts can yield very wide sets.
3. Run fixed first-pass/all-flip comparisons and controlled degradation studies
   without outcome-driven threshold tuning. Report undercoverage rather than
   silently recalibrating on the evaluation labels.
4. Use independently acquired external data only after source/label/overlap
   auditing; freeze all scoring and triage choices before external predictions.

QSART's proposed contribution is to **test** how uncertainty, severity and
quality support an operational dual-budget triage policy and selective CPU
computation. The paper does not validate that policy, QSART model quality or
clinical screening safety. No performance benefit follows merely from adding
conformal sets.

## Verification performed in this session

- `.venv\\Scripts\\python.exe -m pytest tests/test_conformal.py -q`:
  **38 passed**, 0.56 seconds. Synthetic fixtures only; no model performance claim.
- `.venv\\Scripts\\python.exe -m ruff check qsart/uncertainty tests/test_conformal.py`:
  **passed**.
- Tests cover exact/infinite quantiles, inclusive ties and fallback, disconnected
  LAC versus contiguous ordinal sets, exhaustive leave-one-out coverage of a
  fixed finite score population, patient-max calibration, image/patient coverage
  differences, invalid inputs and provenance-bound deterministic JSON reload.

Integration, real-image fitting/evaluation and Docker verification are separate
steps; source presence and these unit tests alone do not complete those steps.
