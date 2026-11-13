# Locked IDRiD external-domain evaluation v1 — 4 November 2026

## Scope and decision lock

This is the first model-outcome evaluation on the audited IDRiD grading release.
Source-origin testimony is confirmed. Images/label distributions and fixed overlap
findings were already inspected, but no external classifier outcome has been seen.
Commit this protocol, exact artifact/config bindings and tested execution sources
BEFORE any external forward pass. Preserve all previous results and production
artifacts. Do not retrain, recalibrate, select a favorable seed, deploy a model,
or tune any threshold/analysis from the observed external outcomes.

Evaluate exactly four already selected EfficientNet-B0 states: the frozen-feature
APTOS baseline and fine-tuning seeds 2026/2027/2028. Seed2026 remains the primary
classifier, as declared before its development results. Primary population is
the previously frozen cross-source-filtered representatives (505); additionally
report full516 and filtered513 observations, without selecting a favorable view.
These are fixed external-domain diagnostics, not a patient-independence guarantee,
clinical validation, exact base-paper reproduction or a triage-utility experiment.

## Inputs and inference

Reuse the original immutable data audit, source receipt, reviewed population table
and model/calibration reports. Bind their SHA256 and every checkpoint/score input
in configs/idrid_external_v1.json. Preserve source DR labels, original partitions,
all ambiguous-component flags and existing membership choices. No APTOS internal
reserved-test image is inferred; no IDRiD image enters any fitting operation.

Use source-compatible one-pass fundus crop/RGB/letterbox224 and ImageNet mean/std,
deterministic eval/inference mode, float32 CPU, four threads, channels-last,
batch16 and at most two image-cache workers. Validate every external image against
the original SHA before caching canonical uint8 pixels. The dedicated external
cache accepts only external_holdout, training-ineligible rows. It is not a path
around the generic training-reader guard. Hash cached pixels and source bindings.

Before external inference, verify each loaded predictor on ten development images:
the first two saved selection rows per grade, same source IDs for all four models.
Use the existing bound canonical fitting/selection cache and saved selection
logits. All predicted grades must agree; maximum raw probability delta <=1e-5.
This is a regression check against existing development outcomes, not new selection.
Fail closed on mismatches. Retain checkpoint/state hashes before/after prediction.
No stochastic training operation or weight/buffer adaptation is allowed.

## Frozen probability and uncertainty transforms

Reuse each model's already fitted APTOS temperature exactly; do not call a
temperature optimizer. Reuse all14 existing conformal artifacts per model exactly:
LAC, deterministic ordinal and five randomized ordinal seeds2026–2030 at alpha
.10/.05. All thresholds were fitted on254 separate APTOS representatives, after
independent-role262-image temperature fitting. No IDRiD recalibration occurs.

For uniform seed s, recreate SeedSequence(s).spawn(3): source calibration used
child0 and development selection child1; external observations use child2.
Generate516 uniforms in the original external manifest order, shared across models
and alpha. Retain all five replicates and separate training seeds from uniforms.
Keep raw sets and common empty-set argmax enlargement separate for every method.
There are56 model/method/alpha/uniform comparisons,57,792 raw/fallback individual
sets, with the same saved predictions summarized on all three populations.

## Outcomes, annotation sensitivity and uncertainty

For every model/population report raw and source-temperature grading: accuracy,
macro-F1, QWK, confusion matrix, every grade support/precision/recall, NLL, ECE10,
Brier and referable AUROC/AUPRC. Report conditional mean/sample SD across the three
fine-tuning seeds, never treating them as independent cohorts or choosing a winner.
The source baseline remains a comparator, not a fourth training-seed replicate.

For every raw/fallback prediction-set view report empirical inclusion, all five
grade denominators/coverage, size distribution/mean, empties and contiguity. No
marginal, grade-conditional or patient guarantee transfers under external shift.

Four original fingerprint components contain conflicting source DR grades. For
each observation let A be ALL DR labels observed in its original component. In
addition to as-provided labels, report an annotation-sensitivity envelope for any
predicted set S: lower event A subset S, upper event A intersects S. A singleton
prediction gives the analogous accuracy envelope. These are hypothetical bounds
assuming a grade in A is appropriate for that observation; fingerprint components
may join distinct views and A need not contain the clinical truth. They are not
confidence intervals, corrected ground truth or guaranteed clinical bounds.
Report the unambiguous-component stratum separately as a label-dependent diagnostic,
never substitute it for the fixed main population or claim population-wide validity.
Do not invent macro-F1 sensitivity bounds or arbitrarily correct source labels.

For each fine-tuned model versus the fixed baseline, compute descriptive paired
fingerprint-component bootstrap accuracy/macro-F1 differences on all three views:
2,000 resamples, seed20261105,95% percentile intervals, fixed predictions, complete
groups and duplicate draws retained. No refitting, patient-independence assumption
or multiplicity-adjusted significance claim. Nine paired comparisons remain visible.

## Preservation, reporting and consumed status

Save private per-image logits, uniforms, sets, cache and manifests under ignored
artifacts/idrid_external_v1. Public aggregate reports, figures and uncertainty
tables go under reports/idrid_external_v1 and refuse overwrite.
Exact-source --resume may reuse verified complete model exports and canonical
cache under the identical execution lock. Preserve incomplete temporary exports
and recompute only the incomplete predictor. Never mix source/config/input states,
overwrite completed evidence or describe repeated external reuse as fresh.
Publish source,
artifact, input and result hashes and explicitly retain unfavorable findings.
Independently reconstruct all grading/set/ambiguity/bootstrap outcomes from saved
logits/sets and original labels; verify preserved original bytes. Run relevant
tests, Ruff and actual native API/UI checks. Container/browser verification remain
separate from native component execution.

Once predictions are generated, mark IDRiD outcomes consumed for this locked
evaluation. Future reuse is disclosed and may not be called fresh independent
confirmation or used silently for final model/policy selection. Official/internal
APTOS reserved-test inference remains untouched. Overall scientific acceptance
and useful validated triage remain IN PROGRESS; formal manuscript stays deferred.
