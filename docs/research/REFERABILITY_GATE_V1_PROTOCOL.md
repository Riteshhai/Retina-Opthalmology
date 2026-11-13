# Direct referability prediction sets for guarded development triage

The previous five-grade ordinal gate required specialist review for every
selection image. This diagnostic tests whether predicting sets over the
triage endpoint directly changes referral eligibility, actual workload and
observed misses. It follows inspection of the earlier development results.
[Prior ordinal diagnostic](ORDINAL_STUDY_V1_RESULTS.md) and
[guarded policy comparison](POLICY_COMPARISON_V1_RESULTS.md).

## Fixed inputs

Use the selected DeepDRiD model and bound first-pass probabilities from the
ordinal and controlled-degradation studies. The conformal calibration cohort
contains 84 images from 21 patients. The stored first-pass temperature was
fitted on a separate 84 images from 21 other patients. The clean selection
cohort contains 168 images from 42 other patients and previously informed
model selection. Verify the input hashes, sidecar split identities, cohort
counts and exact score ordering. Use the saved raw five-grade probabilities
with the independent temperature for all set calculations. Use the existing
clean first-pass production probabilities, quality and normalized entropy
for the separate action guards and allocation costs. Fit no model or
temperature. The consumed confirmation, reserved tests and external
cohorts supply no inputs or outcomes.

## Prediction sets

Define non-referable as source grades 0–1 and referable as grades 2–4,
following the existing QSART research endpoint. Evaluate three fixed
methods at alpha 0.10 and 0.05.

1. **Projected ordinal** projects the existing five-grade deterministic
   greedy ordinal set into the two endpoint labels.
2. **Projected LAC** projects the five-grade least ambiguous classification
   set into the same two endpoint labels.
3. **Binary LAC** sums temperature-scaled probabilities within each endpoint
   before calibrating the two-class score. The candidate score is one minus
   its endpoint probability.

Calibrate each threshold from the maximum true-label nonconformity score
across each calibration patient's views. Use order statistic
`ceil((21 + 1) × (1 − alpha))` with inclusive ties. Evaluate the raw inverted
sets, including empty sets. The deployed ordinal baseline at alpha 0.10
must reproduce its saved five-grade sets before projection.
[Least ambiguous classification](https://arxiv.org/abs/1609.00451).

Report image inclusion, class inclusion and patient all-views inclusion
with denominators, mean binary set size, empty count, and the count of
non-referable singletons. The patient-level calibration unit accounts for
correlated views in the score. An exchangeable-patient guarantee would
require comparable independent patient clusters and sampling. This reused
development cohort does not establish that guarantee or a safety bound
conditional on automatic clearance.

## Fixed routing and reporting

Use the current v2 pointwise guards and dual-budget allocator with the
default risk, severe-risk, quality, entropy and cost settings. For this
research adapter, expand binary labels to their member grades when passing
sets to the existing allocator. A nonempty set permits consideration for
automatic clearance only if every member is below grade 2. Empty sets and
sets that contain any referable label require review. Record pointwise
recommendations separately from budgeted actions.

Evaluate the whole 168-image selection cohort at the four fixed
review/recapture budget pairs `(.30, .20)`, `(.60, .40)`, `(.90, .40)` and
`(1.00, .40)`. Apply independent integer floors, yielding capacity pairs
`(50, 33)`, `(100, 67)`, `(151, 67)` and `(168, 67)`. Retain all 24
scenarios and record explicit overload with unavailable action metrics
when mandatory deferrals exceed capacity.

For successful allocations, report automatic, recapture and review
counts, their true-grade breakdowns, referable and severe automatic
misses, severe recaptures, and the existing observed research proxy per
image. That proxy adds grade-weighted referable automatic misses,
grade-weighted severe recaptures, 0.25 per review and 0.20 per recapture,
using grade weights `(0, 0, 3, 6, 10)`. Report the same action outcomes
for pointwise recommendations without treating those as capacity-feasible.

Run an independent reconstruction of calibration, set membership,
coverage, guard eligibility, capacities and allocation evidence. Keep
row-level predictions, sets, identifiers and actions in ignored local
storage. Publish aggregate findings, including overload and adverse
outcomes. Keep all methods and both alphas in the report without choosing
a winner. The experiment does not authorize deployment or imply a
validated clinical safety threshold.
