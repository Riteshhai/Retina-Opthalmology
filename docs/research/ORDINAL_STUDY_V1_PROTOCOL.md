# Ordinal score and calibration-unit study — declared 22 October 2026

This development study follows the inspected all-five-grade ordinal outcome. It
tests fixed explanations for that outcome rather than selecting a replacement
from favorable results. It is not fresh confirmation or exact reproduction of
the complete APTOS/base-paper experiment. No deployed checkpoint, temperature,
conformal sidecar, gate or action policy will be replaced by this study.

## Frozen inputs

Use development manifest SHA256
`45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d`.
Use the existing selected DR/quality/base-temperature artifacts and conformal
sidecar whose exact SHA256 values are recorded in ROBUSTNESS_V1_PROTOCOL.md.
Only first-pass inference is studied. Reuse the existing independent first-pass
DR temperature 1.2517522400 at its full stored precision; do not refit it.

Keep the same calibration subdivision: order 42 calibration patients by SHA256
of `spec-v1-conformal-20261020:<subject_id>`, lexical ID tie break; first 21 are
temperature fitting, remaining 21 / 84 images are conformal scoring. Do not use
temperature-fitting observations in score calibration. Verify the existing split
fingerprints, disjointness and original image hashes. Infer only the 84 conformal
images using the selected DR model, original deterministic evaluation pipeline,
batch16, CPU2 threads. Fit no classifier or temperature parameters.

Reuse clean first-pass raw DR softmax probabilities for all 168 selection images
/ 42 patients from the preserved robustness cache. Verify its checksum against
the public robustness report and bind its population/order, manifest and model
identities. Recheck selected original image bytes. Transform raw scores with the
same independent temperature as conformal calibration. Confirmation, official
test and external IDRiD observations are excluded.

## Fixed comparisons and randomization

Compare LAC, deterministic greedy ordinal, and randomized greedy ordinal scores
at alpha .10 and .05, without selecting a winning method/alpha/seed. Greedy
ordering starts at the lowest modal grade and adds the higher-probability
adjacent grade, choosing the higher neighbor on ties. For candidate y with
inclusive cumulative mass M_y, the randomized score is M_y - U*p_y. One U is
shared across candidate labels of the same observation, preserving nested
contiguous prefixes. This independently implements the published mathematical
randomization; other paper dataset/training/quantile details differ.

Use five fixed randomization seeds: 2026, 2027, 2028, 2029, 2030. For each seed,
instantiate NumPy PCG64 via default_rng(seed), draw 84 uniforms in fixed conformal
manifest order, then 168 uniforms in fixed selection order. These are paired
randomization replicates of one classifier/population, not independent training
runs or independent clinical cohorts. Report all five outcomes and mean/min/max;
do not select the most favorable seed. The pure score module takes explicit
uniform vectors and has no implicit RNG.

Primary calibration uses the maximum true-label score per patient: 21 independent
units, exact kth order statistic ranks 20 and 21 at alpha .10/.05. A separate
pooled-image diagnostic uses all 84 image scores. Those views are correlated;
pooled results carry no iid-image coverage guarantee and cannot replace the
patient-all-views target. Do not interpret narrower sets under a changed target
as an improvement on the same guarantee.

Use inclusive score <= threshold and conservative modal fallback for empty sets.
Report raw empty-set counts before fallback separately. Report deterministic
true-label terminal-order fractions, score-at-one counts, and patient-max score
distribution to diagnose the atom at one. Reproduce existing patient-max LAC and
deterministic thresholds within 1e-6 and original selection sets exactly before
accepting the new comparison. Record any discrepancy instead of silently changing
the old artifact. New randomized artifacts use their own research-only schema.

## Outcomes and interpretation

For every fixed row, report image/per-grade/patient-all-views coverage, mean size,
full-size fraction, empty counts, contiguity and grade denominators. Report the
fraction of sets wholly below referable grade2 and unchanged guarded pointwise
action counts, preserving all referability/severity/quality/uncertainty gates.
These counts do not establish cohort-budget feasibility or completed review.
No policy-tuning or automatic-triage benefit follows from a smaller mean set.

Report all five seeds for both calibration units and both alpha values. No formal
coverage claim follows from already inspected development data, stratified patient
sampling or randomization. Method exploration after seeing a failure is disclosed.
Retain broad sets, undercoverage and zero automatic clearance if observed.

Private probabilities, group identifiers, uniforms and prediction sets stay under
ignored artifacts/ordinal_study_v1. Public aggregates and plots go under
reports/ordinal_study_v1. Refuse existing outputs. Bind protocol, execution source,
packages, manifests, inputs, fixed model artifacts and private output hashes.
