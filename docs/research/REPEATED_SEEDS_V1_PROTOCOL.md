# Cache-independent repeated training seeds — declared 22 October 2026

This study estimates development training variability after fixing a discovered
RNG dependency. Historical runs remain genuine measurements of their recorded
procedure. They are not overwritten or mixed into this new three-seed summary.
No candidate will be selected for deployment from these results, and no consumed
confirmation, official test or external images will be predicted.

## Reproducibility change

The previous trainer seeds Torch before frozen-feature extraction. Creating a
DataLoader iterator on a cache miss can advance global RNG state, while a cache
hit does not. Linear heads initialize afterward. Explicitly reseed immediately
before constructing/fitting linear heads, after feature extraction/loading, and
record that initialization rule in metadata. Verify cache-miss/cache-hit head
results with software fixtures. This change defines a new training procedure;
do not retroactively claim that prior seed2026 weights reproduce under it.

## Fixed design

Run seeds 2026, 2027 and 2028 under this same new procedure for both DR and the
four-head quality model (overall, clarity, field definition, artifact). Six runs,
all MobileNetV3-small with the original official ImageNet weights. Use the frozen
development manifest SHA256
`45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d`.
Fit on 848 images / 212 patients; checkpoint-select on 168 images / 42 patients.
The 168 calibration images / 42 patients remain unused for fitting, selection,
predictions, calibration or model comparison in this study.

Copy configs/development_v1.yaml, changing only the seed for the three named
config files. Keep frozen backbone/BatchNorm, original+horizontal-flip training
features, 224px, extraction batch16, classifier batch64, weighted cross-entropy,
AdamW lr .01/weight decay .0001, max100 epochs, patience15, CPU2 threads,
num_workers2. Checkpoint selection remains macro-F1 then unweighted NLL, with
the existing multihead quality selection rule. Do not resume runs.

Use the source-valid feature-cache key and existing official weight file. The
changed source requires one fresh extraction; never rebind or rename an old
cache to bypass provenance validation. Subsequent seeds/tasks may share the
new frozen features. Preserve input/cache/weight/source hashes and actual
update/epoch counts for every run. Record CPU elapsed times as execution facts,
not a matched architecture or deployment-speed comparison.

## Fixed analysis and preservation

Report every seed's DR accuracy, macro-F1, QWK, NLL, ECE, Brier and referable
sensitivity/AUROC/AUPRC. Use uncalibrated selection probabilities for this
training-variability study. Define referable sensitivity using argmax grade>=2,
consistent with the historical model-comparison table. Report quality and three
natural-defect head accuracy/macro-F1/NLL and class supports. Use mean and sample
standard deviation across the three training seeds for numeric outcomes, with
no best-seed selection or inferential confidence claim from three runs.

Do not refit temperatures, fit conformal sidecars, run a new confirmation or switch
the currently selected deployment. Results are selection diagnostics from one
population and one split; they do not measure cohort/domain variation. Historical
seed2026 results may be shown separately only with the procedural difference
explicit, never as a paired improvement or as part of the three-seed summary.

Private models and individual predictions stay under checkpoints/repeated_seeds_v1
and ignored artifacts. Public aggregate reports/plots go under
reports/repeated_seeds_v1. Refuse existing run outputs and preserve failures.
Commit this protocol and source before starting genuine training. Formal paper
drafting and external validation remain separate work.
