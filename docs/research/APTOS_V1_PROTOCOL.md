# APTOS development v1 — declared 3 November 2026

This study follows the user's confirmation that the locally downloaded archive
comes from the official APTOS 2019 Kaggle competition. It extends the completed
DeepDRiD prototype with a genuine APTOS classifier study. It is a CPU frozen-feature
baseline, not an exact reproduction of the base paper's full fine-tuning protocol.
No model or uncertainty setting will be promoted to the application by this study.
Formal manuscript drafting remains deferred.

## Source, integrity and exclusions

The discovered archive has 10,215,289,875 bytes and 3,662 training-image members.
These inventory facts were inspected before this declaration; no model outcomes
or new source-grade distribution informed the plan. The user confirmed the official
Kaggle source on 3 November. Record the actual archive and annotation SHA256,
streaming ZIP CRC verification and extraction/source hashes. Source attribution
and format checks do not independently authenticate a Kaggle-published checksum.

Preserve the original archive. Extract only train.csv and its train_images PNGs
into the project's ignored data tree. Do not extract or infer on competition test
images. Reject traversal, symlinks, encryption, duplicate case-insensitive member
names, invalid identifiers, missing/extra training images and unreadable images.
Validate original integer grades 0–4; do not invent patient, eye or quality labels.
Hash, decode and fingerprint every labeled image before freezing a manifest.

Use SHA256 or central-green DCT pHash distance at most 2 to form connected
components, as in the existing adapter. Preserve strict conflict rejection as its
default. This study explicitly permits one conservative alternative: quarantine
the ENTIRE component if any source grades disagree. Keep every original grade
and private exclusion row; never vote, relabel, select the favorable image or
weaken the distance threshold. Report excluded images/components and grade counts.
Any unreadable image or other schema/integrity failure stops preparation. These
rules are declared before inspecting the actual duplicate outcomes. Quarantining
fingerprint candidates may exclude false positives and changes the population.

## Fixed partitions and independence

Assign retained label-consistent components with the existing grade-stratified
hash allocator: seed 20261020 and group fractions train .60, validation .15,
calibration .15, internal test .10. Preserve the existing rounding rule and report
actual image/group proportions. Every detected component stays in one partition.
Freeze both the complete private manifest and a separate development manifest
containing only train/validation/calibration rows. The trainer must reject test
rows. The internal reserved test is distinct from inaccessible competition labels.
Neither test population is used for fitting, selection, calibration or this report.

Standard APTOS annotations provide no authoritative patient identity. Fingerprint
components are not patients and cannot prove independence of people or eyes.
Never report patient counts, patient-disjoint splits or patient coverage for APTOS.

Within calibration, sort component IDs by SHA256 of
`aptos-v1-temperature-20261103:<split_group_id>` with lexical tie break. The first
floor(number of groups / 2) fit temperature, the rest fit conformal thresholds.
All members of a component have the same role. Choose one representative per
component by minimum SHA256 of
`aptos-v1-representative-20261103:<image_id>` with lexical tie break. Record this
choice privately before model fitting. Temperature fitting may use all images in
its role; conformal score fitting uses only representatives in its disjoint role.
No calibration observation enters classifier fitting or checkpoint selection.

## Two genuine CPU candidates

Train ImageNet-initialized EfficientNet-B0 and MobileNetV3-small with the existing
cache-independent linear-head trainer. Use the same fixed settings for both:
224px crop/RGB/ImageNet normalization, seed 2026, extraction batch 16, head batch
64, maximum 100 epochs, patience 15, AdamW learning rate .01, weight decay .0001,
training-only horizontal-flip feature views, class-weighted cross entropy,
two CPU threads and two loader workers. Freeze backbone weights and BatchNorm
statistics. Use only locally preserved official pretrained files with verified
hash prefixes; no downloads or random-backbone substitute.

Select each checkpoint by validation macro-F1, breaking ties by lower unweighted
NLL. Select the study candidate by the same two metrics, then EfficientNet-B0
before MobileNet in an exact tie. Retain both complete histories, selected/last
artifacts, all validation predictions and every outcome. The backbone comparison
is development selection evidence; no paired improvement over DeepDRiD is claimed.
Record source commit, source/config/input/cache/weight/artifact hashes and actual
epochs, optimizer updates, model sizes and elapsed durations. Durations on this
shared machine do not establish architectural speed superiority.

## Independent calibration and conformal diagnostics

Choose the classifier before fitting uncertainty methods. Obtain its actual
first-pass logits on calibration and validation only. Fit one temperature on the
temperature-role images and apply it identically to score calibration and
validation. Do not fit probability transformations on conformal scores or use
coverage/size outcomes to choose the classifier, alpha, method or seed.

At both alpha .10 and .05 fit conventional LAC, deterministic adjacent ordinal,
and explicit randomized ordinal for all five seeds 2026–2030. Use independent
calibration/evaluation random streams for each seed and retain all 14 comparisons.
Reuse each seed's uniforms across the two alpha values. Do not choose a favorable
seed or report randomization variability as model-training variability.

Conformal fitting uses one representative image per fingerprint component, with
the image-unit artifact schema. Its metadata must state this sampling definition;
do not call these components patients or serialize patient-max scope. Report
empirical results both on all selection images and on one representative per
selection component: image inclusion, mean/set-size distribution, all-five sets,
empty sets, contiguous sets and coverage/support within every grade. Report raw
empty sets and the common nonempty argmax fallback explicitly. Keep method and
fallback definitions comparable; randomization is an independent research module.
Grade stratification, quarantine, unknown patient overlap and repeat views mean
these diagnostics do not establish iid-image, patient, conditional or external
coverage guarantees. Selection data are already used to choose the classifier.

Retain raw/calibrated grading accuracy, macro-F1, QWK, confusion matrix, per-grade
support, referable AUROC/AUPRC, NLL, ECE and Brier. Calibration-fit diagnostics are
not independent test results. No APTOS quality metrics or quality-aware triage
claims are possible from absent quality annotations. Existing DeepDRiD quality
predictions are not labeled APTOS quality supervision.

## Preservation, verification and reporting

Commit the protocol and execution source before real training/calibration. Refuse
overwriting completed experiments. Keep data, identifiers, manifests, exclusions,
per-image scores and trained weights ignored locally. Public reports contain only
aggregate results and hashes, with exact-byte Git attributes. Preserve all existing
DeepDRiD experiments, selected models, temperatures, sidecar and action defaults.

Test archive/path/integrity failures, conflict quarantine without relabeling,
component separation, held-out rejection, temperature/conformal disjointness,
representative selection and report privacy. Independently recompute manifest
grouping/partition rules, grading metrics, checkpoint choice, quantiles and sets
from saved artifacts. Reuse existing verified score primitives rather than silently
altering production algorithms. Run relevant pytest/Ruff and the complete genuine
model integration suite after integration. Expose the report with dataset APTOS_2019
and explicit development scope; verify the API/UI with real saved aggregates.

Current Docker availability must be checked again; historical execution does not
verify a new container. Report unavailable access or environmental blockers
precisely. A release candidate remains partial while mandatory research or
deployment acceptance criteria are unmet, regardless of the calendar deadline.
