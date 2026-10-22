# Specification-v1 alignment experiment — declared 20 October 2026

This protocol precedes the new conformal fitting, score generation, EfficientNet
candidate and non-referable policy experiment. It does not replace or relabel the
consumed v1 confirmation. All new results in this round are development evidence.
Official test and external IDRiD remain unused. APTOS requires official local data
and access terms; a DeepDRiD run is not an APTOS/base-paper reproduction.

## Fixed inputs and scope

Reuse development_v1.csv, SHA256
45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d.
Its fitting/selection/calibration partitions contain 848/168/168 images from
212/42/42 patients. Keep selected real DR and quality checkpoints and the v1
temperature artifact unchanged. Existing numerical prediction reports remain
valid for those artifacts, but old triage metrics do not describe the new action.

## Independent conformal sidecar

Order the 42 calibration patient groups by SHA256 of
`spec-v1-conformal-20261020:<subject_id>`, with lexical ID as a deterministic tie
break. The first 21 patients / 84 images fit a new DR temperature independently
for first-pass logits and original/flip mean logits. The other 21 patients / 84
images fit conformal scores. These model parameters/checkpoints were selected on
the separate selection partition, not either calibration subgroup. Do not use
the old all-168 fitted DR temperature to fit conformal scores.

Evaluate LAC and paper-inspired deterministic ordinal-greedy scores at alpha .10
and .05. Use each patient's maximum score as one calibration unit. The ordinal
construction and tie policy are documented against the primary paper and authors'
code; this is not exact randomized replication. Display CP's independently
temperature-transformed probability source explicitly alongside unchanged v1
grading probabilities. CP does not retroactively validate the v1 model.

Separate first-pass/all-flip paths; conditional mixed TTA uses the corresponding
path artifact but receives no unconditional/selection-conditional coverage claim.
Patient clustering, original severity-stratified sampling and domain shift limit
formal guarantees. Report empirical marginal/per-grade/all-views coverage, set
sizes and absent strata on selection data as development diagnostics only.

## Current policy contract and simulator

Use AUTO_NONREFERABLE, RECAPTURE and SPECIALIST_REVIEW under triage_v2, with
explicit default research gates: referable grade2, referable risk .5, severe
grade3/risk .25, minimum good-quality probability .5 and entropy .75. Predicted
referable, high-risk or referable-containing conformal sets mandate review;
remaining poor-quality cases defer to review/recapture. Never force automatic
non-referable clearance to satisfy a capacity. All thresholds/costs are proxy
assumptions, without clinical validation.

Run all eight guarded strategies, showing no-deferral raw classification as a
distinct comparator. Separate floor review/recapture capacities. Preserve every
overloaded budget case, with no invented actions or favorable zero-miss metric.
The interactive simulator accepts only cached selection data for dataset-based
exploration; reject holdout/test population caches. It is not a final evaluation.
Compute profiles are predefined first-pass, 25%-budget TTA and all-image TTA;
the simulator must disclose their precomputed activation/capacity scope.

## EfficientNet candidate and APTOS preparation

Add a backward-compatible EfficientNet-B0 implementation and independently train
a new frozen ImageNet feature/head candidate on the same 848 fitting images,
168 selection images, 224px and configs/development_v1.yaml settings, seed2026.
Use max100epochs/patience15 and selection macro-F1 then unweighted NLL. It is an
additional development candidate, not a blind confirmation comparison. Do not
replace the running model or reuse its reported accuracy for the new candidate.
Record actual training/weight/source hashes and timings, with shared-machine
training time unfit for a speed claim. No confirmation inference is authorized
for this candidate under a fresh-confirmation label.

APTOS adapters may parse local id_code/diagnosis images, validate bytes and use
connected duplicate groups. No patient/quality identifier is fabricated. Default
proposed patient-unknown split is 60/15/15/10, seed20261020; no actual split occurs
until official source data are available. Reserve resulting test data untouched.

## Verification

Use generated fixtures for mathematical/endpoint guards and real selected models
for calibration, score-cache generation, single/batch/conformal/simulator requests.
Run Ruff/pytest and rebuild/restart Docker. Bind sidecars, simulation caches and
reported experiments to model, calibration, source, manifest and protocol hashes.
Do not claim prospective benefit, clinical safety, external validation or policy
superiority from this alignment round. Formal manuscript drafting stays deferred.
