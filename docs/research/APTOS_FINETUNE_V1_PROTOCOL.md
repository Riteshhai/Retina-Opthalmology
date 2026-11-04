# APTOS parameter fine-tuning v1 — declared 4 November 2026

The frozen-feature study selected EfficientNet-B0 at macro-F1 .60334, with grade-3
recall only 6/26. This follow-up tests backbone adaptation and conditional training
variability. Those development findings informed this plan; it is not confirmatory
research. Preserve the completed study and all production artifacts. No automatic
deployment change or formal manuscript drafting follows from this experiment.

## Fixed cohort and relocated source

Reuse the original retained 3,514-image cohort, exclusions and component partitions.
The original development manifest SHA256 is
`681799c7ee5f04e988dc607f6cf296fd882c5f55f2603dcd0a2ccd8148951fab`;
the full manifest is `1d79240d22237d1934328899d3d6624cd41f6294a15d158a3f62eacf5e7684ed`.
The original study report SHA256 is
`994a8e78148d6c16d601d61d8774e0d36070b653ebf5549dec093ecbad4924ec`.
Train/selection/calibration counts remain 2,115/521/527. The 351 internal reserved
test images and competition test are not used for inference or outcome evaluation.
Source labels were already used for the declared original grouping/stratification.

Resolve the active source through QSART_APTOS_ROOT, honoring the user's relocated
folder. Build a private runtime development manifest by mapping original image
identities into its train_images directory; verify every used PNG against original
SHA256 and retain labels/splits/groups exactly. Do not rewrite the original frozen
manifests or regroup images. Bind original/runtime manifests and active input bytes.
Reject missing/changed images, role mismatches or any test/holdout training row.
No new quality, patient or eye annotations are invented. Fingerprint components
are not patients; quarantine and unknown related views limit all conclusions.

## Fixed model and training replicates

Every run starts from the SAME prior selected APTOS EfficientNet-B0 checkpoint,
SHA256 `52574e61db4b68308348aab3794bcc64c55f6d47b0d02932955286c35d190407`.
Its backbone is the unchanged official ImageNet state and its five-way head is
already fitted using the original selection protocol. No checkpoint or seed is
chosen after observing this follow-up's results. Seeds 2026, 2027 and 2028 govern
fine-tuning order, horizontal flips and inherited EfficientNet stochastic depth.
Their sample SD is conditional on this shared initialization and split, not
full-pipeline uncertainty or independent classifier-head initialization variability.

Make all backbone and head LEARNABLE PARAMETERS trainable. Keep BatchNorm running
means/variances/num_batches_tracked fixed through eval mode on those modules while
retaining gradients for their affine parameters. Keep the rest of the model in
training mode, including inherited stochastic-depth behavior. Verify changed
backbone parameters and unchanged BatchNorm running buffers in saved results.
This is parameter fine-tuning with frozen BatchNorm statistics, not the base
paper's exact 384px end-to-end training procedure.

For every seed use the same fixed settings: original one-pass fundus crop/RGB
224px preprocessing and ImageNet normalization; batch 16; maximum 12 epochs;
patience 4; AdamW weight decay .0001; backbone initial learning rate .0001 and
head .0003; CosineAnnealingLR T_max=12, eta_min=.000001; two CPU threads. Use
training-only per-image Bernoulli(.5) horizontal flips and class-weighted cross
entropy with weights N/(5*n_class) from the ORIGINAL training records. No sampling
or severity-loss hyperparameter is adjusted after results. Each epoch uses every
training image once. Validation is deterministic, without augmentation or gradients.

Select a checkpoint among TRAINED epochs by validation macro-F1, then lower
unweighted NLL; stop after four consecutive non-improving epochs. An unchanged
epoch-zero warm start remains the separately reported baseline, never a newly
fine-tuned checkpoint. Retain all histories, selected/last weights, actual optimizer
counts and every outcome, including degradation. Seed2026 is the predeclared
primary model regardless of which seed scores best. Never deploy a favorable seed.

## Canonical image cache and reproducibility

Cache deterministic cropped/resized RGB pixels as uint8, separately from normalized
model tensors. Build the training/selection cache from those partitions only.
Verify equality with ordinary image_to_tensor preprocessing in software tests and
retain actual image/cache/source/content hashes. Cache calibration images only
for later post-hoc inference in a separate accepted scope. Cached images must
never receive a second geometric crop. Random augmentation occurs AFTER cache
reading, so no permanently augmented sample becomes a validation observation.

Construct/read caches and load the warm start BEFORE reseeding all relevant RNGs.
Use explicit reproducible batch/augmentation generators and seed Torch's global
RNG for stochastic depth. Cache existence must not alter ordering, augmentation
or learned weights. Record package/config/source/input hashes and clean source
commit before execution. Checkpoint resumability, if supported, must restore full
optimizer/scheduler/epoch/patience and all RNG states with identical provenance;
never mix changed configurations or claim interrupted partial epochs as completed.

## Development comparisons and independent calibration roles

Retain all three seeds' raw selection grading metrics, confusion matrices,
per-grade recall/support, NLL/ECE10/Brier, referable AUROC/AUPRC and epochs/updates.
Compare each to the same frozen-feature baseline on the exact same 521 images.
Report all seed values plus mean/sample SD. Do not interpret the three runs as
independent test cohorts or a prospective performance estimate.

For accuracy and macro-F1, compute descriptive paired fingerprint-component
percentile bootstrap differences (fine-tuned minus baseline), 1,000 resamples,
seed20261104. Retain all views in each sampled component, both fixed predictions
and duplicate draws. No retraining/reselection occurs in resamples; unobserved
patient correlations, selection reuse, three comparisons and no multiplicity
adjustment limit inference. These intervals do not establish held-out superiority.

For EACH trained seed independently, retain the original temperature-role 262
images and disjoint 254 conformal representatives, using the unchanged role and
representative hashes. Fit temperature solely in its role. Reuse the existing
14-row procedure: alpha .10/.05, LAC, deterministic ordinal and all five explicit
randomization seeds 2026–2030 with independent calibration/evaluation streams and
shared uniforms across alpha. There are 42 comparisons across classifier seeds.
Distinguish classifier training seeds from conformal randomization seeds.

Report RAW sets and common argmax fallback separately, all 521 selection images
and 507 selection representatives, every grade's coverage/support, raw empties,
size distribution, contiguity and all-five counts. No alpha, randomization seed,
temperature procedure or method is selected to hide severe-grade failures.
Selection images already selected checkpoints; these remain empirical development
diagnostics. Image-unit artifacts describe representative calibration and never
claim patient, grade-conditional, external or clinical guarantees.

## Execution and preservation

Commit this protocol/config and tested execution sources before actual fine-tuning.
Generated-tensor software compute probes may precede execution; label them clearly
and never report them as retinal model evidence. No source/parameter edits during
the three-run study. Private inputs/caches/checkpoints/logits/sets remain ignored
under artifacts/aptos_finetune_v1 and checkpoints/aptos_finetune_v1. Public reports
are aggregate-only under reports/aptos_finetune_v1 and refuse overwrite.

Independently audit checkpoint selection, weight/BatchNorm changes, source/config/
cache/image bindings, paired comparisons and uncertainty outputs. Preserve all
original APTOS and DeepDRiD artifacts byte-for-byte. Expose the saved report in
the research explorer with development scope and verify actual native HTTP/UI
components. Run meaningful tests and Ruff; record actual outcomes and remaining
blocks. New container verification remains separate from historical Docker proof.
