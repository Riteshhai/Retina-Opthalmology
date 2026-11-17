# Retrospective DeepDRiD paired-view acquisition

Lock this protocol and the executable study before scoring. The study asks
whether a model can identify which eyes benefit from an additional recorded
view using only the initial view. DeepDRiD supplies two regular-fundus fields
per eye. The study uses filename suffix `1` as the initial view and suffix `2`
as the additional view. This ordering defines the retrospective experiment.
Capture chronology and the result of requesting a repeat photograph are
unobserved. [DeepDRiD data paper](https://doi.org/10.1016/j.patter.2022.100512) ·
[Prior horizontal-flip study](MATCHED_FLIP_V1_RESULTS.md).

## Frozen populations and predictions

Use the 848 verified Set-A training photos from 212 patients (424 eyes) to
fit the value model. Evaluate on the 168 verified Set-A selection photos
from 42 different patients (84 eyes). Require two eyes and both views for
each patient, with one consistent released grade per eye. Use the frozen
manifest and verify original photo bytes before scoring. The 42 calibration
patients supply neither fitting nor evaluation outcomes for this study.
The previously evaluated Set-B, Set-C and external cohorts remain outside
this experiment. The APTOS test remains reserved.

The DR and quality checkpoints, production temperature and preprocessing
are frozen. Score each photograph with the first-pass pipeline and zero
extra model passes. Use the calibrated five-grade vector of the initial view
when no additional view is acquired. If a recorded second view is selected,
use the arithmetic mean of its five-grade vector and the initial vector.
Keep the existing production triage policy outside this experiment. The
new fusion rule supplies a retrospective classifier result.

## Value model and objective

The reference is the released eye grade. Grades 2–4 are referable and
grades 3–4 are severe under the study definitions. Let `w = (0,0,3,6,10)`.
Before acquisition cost, charge `w[truth]` when the true grade is referable
and the predicted argmax grade is below 2. Charge `.25` when the predicted
grade is at least 2 as a proxy for specialist review work. For each training
eye, compute the signed decrease in this ledger from first-view prediction
to fixed two-view fusion. Allow negative training targets when fusion
introduces a miss or adds review work. These prices and predictions
define a research ledger. The dataset does not observe care outcomes.

Regress the gross ledger decrease on these five initial-view features:
model-estimated poor-quality probability, normalized five-grade entropy,
probability of grade 2 or higher, probability of grade 3 or higher, and
an indicator that the predicted grade is at least 2. Standardize predictors
using the training means and population standard deviations, with zero scales
replaced by one. Fit ridge regression with a fixed unit penalty on the
standardized coefficients and an unpenalized intercept. Freeze the model
before reading selection targets. Acquire only when score exceeds `.20`
in the capped positive-value strategy. Exact-budget strategies rank all
eyes and can use slots with estimated gain below acquisition cost.

Selection is produced from the initial-view features alone. Hidden-view
probabilities and reference grades enter only the retrospective outcome
calculation and the separate fit targets. Classifier fitting and value-model
fitting share training patients. Selection is disjoint from value-model
fitting and was previously used to select the frozen classifier. This is
development evidence with a patient-disjoint value-model evaluation.

## Acquisition comparisons

Use first-view-only and acquire-both as the 0% and 100% anchors. For each
interior extra-view cap of 10%, 25%, 50% or 75%, give each matched method
exactly `floor(cap × 84)` extra views (8, 21, 42 or 63). Rank eyes by predicted
value, estimated poor quality, entropy, or referable probability. Break ties
using the frozen pair order. Use one seeded random permutation (`20261116`)
as a fifth exact-workload method. Also report the capped positive-value rule,
which may leave slots unused. This is one retrospective population with
shared acquisition capacity. It imposes no specialist-review capacity.

Report both the exact number of acquired views and the number of referable
predictions that incur the review proxy. Add `.20` per acquired image to the
eye-level ledger and report its mean as the primary outcome. Report eye
accuracy, missed referable and severe cases, and worst-eye patient misses.
A severe miss is a true grade 3 or 4 predicted below grade 2. For each of
the four interior budgets, compare the value ranking against each of the
four exact-workload alternatives. Report paired candidate-minus-comparator
differences in ledger per eye, missed referable and severe cases per eye,
and grade accuracy. Use 2,000 percentile bootstrap resamples of whole
patients with seed `20261116`, keeping original acquisition assignments
fixed. The intervals describe uncertainty in this fixed-cohort comparison.

Publish all 26 scenarios and 16 paired comparisons, including losses and
negative estimates. Independently reconstruct source linkage, model fit,
rankings, slot counts, outcomes and comparisons from local evidence. Keep
per-photo scores, patient identities and acquisition masks private. Publish
only the aggregate report, audit, a workload figure and an interactive view
of the prespecified budgets. Acquiring another field in these records cannot
establish the benefit of a future acquisition or a real-world retake.
