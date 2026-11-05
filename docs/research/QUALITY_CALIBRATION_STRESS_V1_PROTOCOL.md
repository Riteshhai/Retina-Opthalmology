# Quality calibration and stress v1 — declared 5 November 2026

At declaration, no new calibration-role predictions, temperature fits or stress
responses for these six checkpoints have been generated. Their clean selection
outcomes are already consumed. This is a development follow-up to quality
adaptation v1, not fresh confirmation, clinical validation or deployment selection.

## Fixed models and roles

Freeze all six selected quality-adaptation checkpoints (three training seeds
2026/2027/2028, frozen/adapted arms). The authoritative prior report is
evaluation_v1_1.json, SHA256
b21f0305d3d2c19392d91434b7b4dedaed65c9329029b84939143ca8497f2fd8.
Bind every checkpoint and original selection export through that report and its
seals. No new classifier updates, checkpoint/seed selection or TTA. Primary model
remains `quality_seed2026_adapted`; report all six regardless of outcomes.

Use the unchanged DeepDRiD development manifest SHA256
45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d.
Only decode calibration168 images/42 patients/components and selection168/42/42.
All known views stay together. Fitting848 metadata provide overlap guards only;
no fitting pixels, consumed340-image confirmation, IDRiD or APTOS images enter.

These six models excluded the calibration role from fitting and selection. The
same role was used historically to calibrate older models; this reuse is disclosed.
Selection repeatedly chose and evaluated checkpoints and cannot supply independent
test evidence. No conformal artifact is fitted or changed. The existing production
calibration, models, sidecar, policy and all prior results stay byte-identical.

Clean four-head annotations retain their documented mappings and poor-positive
class0. Calibration/selection supports respectively:

| Head | Calibration supports | Selection supports |
| --- | --- | --- |
| Overall quality | 82,86 | 98,70 |
| Clarity | 5,6,26,56,75 | 5,9,42,61,51 |
| Field definition | 2,6,27,102,31 | 1,6,21,116,24 |
| Artifact | 90,8,33,21,12,4 | 80,6,35,20,20,7 |

## Exactly 24 scalar temperatures

For each fixed model, first reproduce its168 clean selection raw logits against
the prior export (maximum absolute error <=2e-4 and all head argmaxes identical).
Then predict its clean calibration inputs and fit one scalar per head using the
existing `fit_temperature`: unweighted image NLL, float64, LBFGS lr.1/max75/strong
Wolfe, log-temperature clamped to log(.05)..log(20), with T=1 fallback only when
calibration-fit NLL would worsen. No selection labels enter fitting or a raw-versus-
scaled choice. Fit once per model/head; positive scaling preserves class argmaxes.
Neither a classification improvement nor repaired rare-class discrimination is
expected from this transformation. Retain every fit and all adverse outcomes.

On clean calibration (in-sample) and clean selection (separate from temperature
fitting, development-reused), retain raw and scaled accuracy, fixed-universe
macro-F1, NLL, Brier, ten-bin ECE/reliability ([left,right), final right inclusive),
confusion matrices and class precision/recall/F1/support. Overall-quality AUROC,
AUPRC, poor sensitivity/specificity use class0. Keep all four heads and rare strata.

Primary calibration endpoint: selection mean four-head NLL, temperature minus raw,
for the fixed primary model. Report this for all six; per-head NLL/Brier are
secondary paired endpoints (nine intervals/model,54 total). ECE is a point
diagnostic. Report conditional means/sample SDs across all three seeds per arm.

## Seven deterministic model-input conditions

Decode each of336 originals once. Apply the existing fundus crop/letterbox224
once, then independently reuse the seven exact operations from robustness_v1:
clean, Gaussian blur1/2 input pixels, float32 sRGB exposure multipliers.5/1.5
with clipping/rint, JPEG50/20 (subsampling2,optimizeFalse,progressiveFalse).
Bind Pillow/libjpeg versions and every cell's canonical/altered uint8 hash.
Normalize directly without recropping; batch16,CPU2 threads,channels-last.
Calibration has clean inputs only. Evaluate selection seven ways for each model:
7,056 selection image-model observations plus1,008 calibration observations.

Corrupted images have **no new quality or defect annotations**. Do not compute
quality accuracy/NLL/ECE against their original clean annotations. Report score
responses only, for raw and frozen-temperature probabilities:

- Poor-quality probability, positive class0.
- Expected clarity class divided by4 (higher is better clarity).
- Expected field-definition class divided by4 (higher is better definition).
- Expected artifact class divided by5 (higher is more artifact).

These normalized class indices are ordinal-code summaries, not physical or clinical
defect measurements. No response direction is guaranteed; exposure manipulations
are not acquisition simulations. Primary stress endpoints: calibrated poor-score
change from clean under blur2 and JPEG20 for the fixed primary model. All six
nonclean conditions/four responses/two probability states/six models are retained
(288 paired intervals). Clean zero-response controls add48 degenerate intervals.

Also retain all144 adaptation response contrasts: for each training seed, state,
condition and response, compare (adapted condition-clean) minus (frozen condition-
clean). Raw contrasts compare frozen weights; scaled contrasts also include the
separately fitted arm temperatures. No model is chosen from these comparisons.

## Common group intervals and evidence controls

Use2,000 whole split_group_id draws, numpy default_rng20261108,42 selection
components sampled with replacement retaining every view. Reuse one common draw
matrix for all models, heads, transforms and response contrasts. Intervals are
image-weighted paired means with percentile2.5/97.5 bounds, conditioning on frozen
models/temperatures. No classifier or temperature refitting, checkpoint reselection
or multiplicity correction occurs inside bootstrap draws. Retain all486 declared
intervals and48 zero controls, not just primary or favorable results. Training seed
and bootstrap_seed have distinct field names.

Source/config/protocol are tested and committed clean before any new inference or
temperature fitting. Default runner validates inputs only; explicit `--execute`
performs the study and `--execute --resume` requires exact bindings. Atomically
save each inference cell and fitted scalar state; seals reject changed outputs.
Bind all old measured reports, master copies, production artifacts and all six
quality runs. Keep paths, identifiers, per-image outputs, caches and scalar
sidecars local/ignored. Public outputs are aggregate diagnostics and figures.

Independently reconstruct24 fits/objectives, all clean metrics/reliability bins,
all486 intervals/48 zero controls, model/BN invariance, raw clean regressions,
source/cache bytes and sampled geometry/transform pixels. Compare scalar objectives
with an independent bounded optimizer and report gaps without post-hoc refitting.
Verify saved-results API/UI plus the unchanged genuine production pipeline.
Run relevant pytest/Ruff/Compose checks. Container proof requires a working Docker
engine; native/AppTest evidence is labeled separately. Overall QSART acceptance
remains IN PROGRESS and formal manuscript drafting stays deferred.
