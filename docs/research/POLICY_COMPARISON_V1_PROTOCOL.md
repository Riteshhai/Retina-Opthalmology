# Independently tuned guarded policies — declared 22 October 2026

This experiment follows inspected development failures and implements the remaining
independently tuned comparator study. It is development out-of-fold policy evidence,
not fresh end-to-end validation: all 42 selection patients selected the classifier
and have already been inspected. Do not reuse consumed confirmation, official test
or external IDRiD data. No deployed model, temperature, sidecar, guard or default
will be changed by this study. Formal manuscript drafting remains deferred.

## Frozen inputs and allocation semantics

Use all 168 selection images / 42 patients in development_v1.csv, full manifest
SHA256 `45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d`.
Use only the `single` first-pass profile in the existing real development cache,
SHA256 `71013749c8b303104568e8115e0e78f13373ef951e9e5c5bfac1fea9469a0b57`.
Verify its public report binding, model/checkpoint/base-calibration/sidecar hashes,
population order and true labels against the manifest, and original image bytes.
All patient/group identifiers and individual actions stay local and ignored.

Reuse triage_v2's eight guarded wrappers: plain, confidence_only, entropy_only,
severity_only, quality_only, single_budget, naive and dual_budget. They share
risk, severity, entropy, quality and supplied conformal guards; recapture costs
also use severity. These are guarded wrappers, not unrestricted reproductions of
published pure baselines. Plain forbids optional review of eligible cases;
single_budget requires review for poor quality and does not use recapture.

Freeze grade thresholds2/3, referable risk .50, severe risk .25, minimum good
quality .50, entropy .75, severity weights(0,0,3,6,10), recapture decision price
.20, quality weight1 and entropy weight .50. No guard threshold or score weight
is tuned. Recapture is allowed only with a quality indication and never removes
the high-risk review requirement. Separate exact integer capacities apply.

## Prespecified modes, budgets and folds

Three fixed modes, kept separate in all reporting:

- `ordinal`: original deployed deterministic patient-max alpha .10 sets. This is
  the primary operational result, even when every scenario overloads/full-reviews.
- `lac`: existing patient-max alpha .10 LAC sets, a diagnostic uncertainty-layer
  sensitivity study; it is not a deployed replacement.
- `no_cp`: explicitly labeled no-conformal ablation. Other guards remain intact.
  Remove both conformal fields so no hidden fallback restores the ordinal gate.

The prediction-only preflight census is 168 mandatory reviews for ordinal;
147 mandatory reviews plus15 flexible poor-quality cases for LAC;84 plus63 for
no_cp. These counts were inspected without true labels. They limit how much
optional ranking can change and are disclosed rather than hidden.

Use the four fixed (review,recapture) budget pairs:
(.30,.20), (.60,.40), (.90,.40), (1.00,.40). Maximum capacities may be unused.
For N images calculate each cap independently as floor(B*N), with no epsilon.

Sort the42 unique patient IDs by SHA256 of
`policy-comparator-v1-20261022:<subject_id>`, lexical ID tie break, and assign
rank modulo3. Each evaluation fold has14 patients/56 images; the other28 patients
/112 images tune that fold's decision parameter. No patient enters both sides of
a fold. Retain manifest order within each cohort; record SciPy/NumPy versions.
Equal-cost LP optima can depend on solver version or row order, so results do not
claim cross-version assignment identity. Labels never enter allocation.

## Independent tuning and one fixed evaluation ledger

Independently tune every method, mode, budget and training fold over the same
decision-review-price grid (.05,.10,.25,.50,1.00,2.00). This reuses the allocator's
`review_cost` as an action-price parameter; it is not an estimated clinical cost.
Only that allocation parameter varies. The scorer always uses a separate fixed
evaluation configuration with review cost .25 and recapture cost .20.

For every candidate and evaluated fold, use exactly this observed proxy:

`[sum severity weights of referable automatic clearances + sum severity weights
of severe cases sent to recapture + .25*review_count + .20*recapture_count] / N`.

This is an unvalidated research ledger; the prediction-based recapture decision
cost includes expected referable risk, whereas the outcome ledger penalizes severe
recaptures. They are not an identity for expected clinical utility. Tuning the
allocator price must not redefine evaluation prices or severity weights.

Choose minimum tuning-fold proxy, then fewer severe automatic misses, fewer
referable automatic misses, fewer severe recaptures, fewer reviews, more automatic
cases, then fixed grid order. Retain all six candidate outcomes, not only winners.
Evaluate the chosen price on the excluded patient fold. Also evaluate every
untuned wrapper with default decision price .25 as a separate reference.

If all candidates overload on tuning data, mark that fold untunable. If excluded
fold allocation overloads, retain diagnostics and no actions/loss. Never force a
clearance, relax a guard, substitute all-review output beyond capacity or invent a
favorable zero metric. There are3*4*8=96 tuned scenarios and96 default references,
each with3 excluded-fold records, plus96*3*6=1,728 tuning candidates.

## Aggregation and comparisons

Each patient contributes to one excluded fold. Report primary pooled outcomes only
when all3 folds have real actions. Otherwise primary pooled metrics are null,
with explicit untunable/evaluation-overload fold counts and available population
counts; do not silently drop the failed folds. Sum each fold's applied capacities
for pooled accounting, rather than using a new floor on all168 images. Report
actual review/recapture workloads, automatic coverage, referable/severe misses,
referable/severe recaptures, fixed proxy components, selected prices and shortfalls.

Compare tuned dual_budget with all seven other tuned wrappers, and each tuned
method with its own default reference, for every mode/budget. Retain comparisons
that cannot be estimated. Only fully evaluated pairs receive numeric differences.
Label matching of actual review AND recapture counts within every excluded fold;
same budget fractions alone do not imply matched realized workloads. Do not claim
superiority for unequal workloads or full-review/zero-automatic cases.

For fully evaluated pairs, report descriptive paired patient-cluster percentile
intervals for fixed per-image proxy cost, automatic indicator and referable-auto
miss indicator:1,000 resamples,seed20261022. Resample whole patients and retain
both methods' fixed actions. Do not refit thresholds or reallocate in resamples.
These intervals condition on overlapping cross-fit tuning sets and fixed actions;
they are not full cross-validation uncertainty, multiplicity-adjusted inference,
prospective safety or a resource-capacity guarantee on resampled cohorts.

No parameter, budget, CP mode or method is selected for deployment. The production
ordinal failure remains visible regardless of diagnostic ablation outcomes.
No new classifier probabilities or calibration/confirmation predictions are needed.

## Preservation and verification

Commit protocol and implementation source before the actual run. Refuse existing
outputs. Private folds, candidate actions and per-image scores stay under ignored
artifacts/policy_comparison_v1. Public aggregates/plots go to
reports/policy_comparison_v1 with exact source/protocol/input/model/package hashes.
Test fixed-ledger evaluation, patient separation, evaluation-label independence
of actions, overload without metrics/actions, conformal-mode handling, exact caps
and unchanged guard restrictions. Independently recompute the actual outcomes
and selected candidates. Verify the real API/native app after integrating the
saved report; current Docker host failure remains a separate limitation.
