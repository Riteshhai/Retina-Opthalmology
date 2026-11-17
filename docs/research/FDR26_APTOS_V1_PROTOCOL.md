# Paired frozen APTOS and DeepDRiD grading on FDR26 v3

Compare the previously declared primary APTOS EfficientNet-B0 (seed 2026,
selected epoch 4) with saved DeepDRiD scores on the same four FDR26 grading
populations. The earlier DeepDRiD outcomes informed this follow-up. The APTOS
checkpoint is the original primary from the IDRiD protocol, fixed without
using FDR26 scores. This is an exploratory comparison of two frozen pipelines
on a reused external cohort. The pipelines differ in more than training data.
[Prior FDR26 grading](FDR26_GRADING_V1_RESULTS.md) ·
[Prior APTOS model selection](IDRID_EXTERNAL_V1_PROTOCOL.md).

Bind the archive, both readiness receipts, DeepDRiD results and witness,
original APTOS source report, configuration, checkpoint, development scores
and preflight cache. Keep the original 2,849 eligible photos as primary.
Report all three previously fixed sensitivity views. Reuse the saved DeepDRiD
outputs in their exact image order. Before new FDR26 scoring, replay the
ten predeclared APTOS development observations using the existing IDRiD
preflight. Check original archive bytes against the audited hash per photo.

Score APTOS with the source crop and RGB letterbox to 224, ImageNet
normalization, float32 channels-last batches of up to 16 and four CPU
threads. Use the fixed APTOS source temperature 1.7353179819929263 and
the same float64 softmax as in the earlier IDRiD diagnostic. Retain the
prior DeepDRiD inference and calibration values as originally recorded.

Use the released ALFEDIAM codes and the same fixed grade ≥2 proxy, with
probability at least 0.50. Report the original five-way, binary and probability
metrics for both models, plus exact paired counts for both correct, APTOS
only correct, DeepDRiD only correct and neither correct. Use descriptive
image-level outcomes given unverified patient linkage and different grading
rubrics. No inference about clinical referral or care follows from these data.

Independently verify the outcome bindings, four populations, probabilities,
all aggregates and paired counts. Replay the first complete APTOS request
and compare its first ten stored logits at absolute tolerance 1e-6. Keep
image-level evidence local and publish aggregates. Fit no model or threshold
on FDR26 and retain both models' results.
