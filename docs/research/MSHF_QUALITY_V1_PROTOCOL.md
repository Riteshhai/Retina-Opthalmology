# External overall-quality evaluation on MSHF v2

Evaluate the frozen DeepDRiD overall-quality head on the conventional color
fundus photographs in the authors’ MSHF v2 release. MSHF uses a binary
overall-quality judgment from three ophthalmologists, with 0 for poor and 1
for good. This is a comparable endpoint to the DeepDRiD binary overall-quality
head, with source-specific grading criteria. Use only the supplied
overall-quality label. Source disease categories are subgroup descriptors.
[MSHF paper](https://doi.org/10.1038/s41597-023-02188-x).

## Source and cohort

The author deposit is Jin, Gao, Jiang, Wang, Ma, Li and Ye, MSHF v2, Figshare
article 21507564, file 39485878. The 1,136,891,149-byte ZIP has publisher MD5
`58203e6c2e064dafc800f5b83887487b`. The downloaded ZIP matched that digest
and had SHA256 `115def6513bc1516d77ceb4ac2e90a9fcf582e825604c08be9524f5fd6c4a219`.
Read members directly from the archive and keep it and all row-level outputs
local.
[Author deposit](https://doi.org/10.6084/m9.figshare.21507564.v2) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Use the four author source groups under `Original/CFP`: DR-XJU (235),
DR-ZJU (187), Glaucoma (52) and Healthy (26). Join the individual rating
workbook one-to-one by the original filenames. Use the majority of the
three binary overall ratings as the endpoint, following the paper. Inspect
each rating, the archived count and the decode for every row. Preserve the
individual ratings and source groups. The portable, ultra-widefield and
`AI-use` directories are outside this cohort. The paper’s pooled subject
counts differ across its sections. The released rating workbook identifies
images, and patient linkage is unavailable.

The archive preflight found discrepancies between the reconstructed majority,
the summary workbook and the article’s table. In particular, the cached
DR-ZJU summary has all 187 joined rows coded poor and the overall formulas
reference empty cells. Report all workbook disagreements by source. Permit
an exact unambiguous filename stem join for summary workbook auditing when
its filename omits the extension. The summary workbook and article counts
are comparison evidence for the annotation audit. The study endpoint
comes from the complete individual ratings, and this choice was fixed
before running any model prediction.

## Overlap and evaluation populations

Compare each conventional photograph with the other conventional photographs
and the four frozen reference manifests already bound for the IDRiD audit.
These contain the QSART DeepDRiD development and confirmation populations and
the APTOS annotated population, including its quarantined rows. Use byte
SHA256 and the existing central-green retinal pHash with Hamming distance at
most two. Treat pairs as candidates and group internal pairs transitively.

The main population includes all rows outside components with a candidate in
a reference manifest or conflicting overall-quality labels. These whole-component
exclusions are conservative and independent of model scores. The second
population takes the lexicographically first archive member of each
remaining component to check sensitivity to internal repeats. A third
population uses the eligible images with unanimous overall ratings to
check sensitivity to rater disagreement. Report the complete inventory,
candidate and exclusion counts, and results for all three populations,
including each source group. Fingerprints can miss related views and can
flag distinct images. These are image-level external diagnostics with
unresolved patient independence.

## Frozen predictions and metrics

Bind the selected production quality checkpoint and its existing
DeepDRiD calibration artifact. Use the deployed crop, letterbox, RGB
normalization and single-image first-pass forward on CPU. Score the same
logits before and after applying the stored first-pass temperature.
All model, temperature and decision choices come from the existing
DeepDRiD pipeline. Evaluate the fixed quality flag `P(poor) > 0.50`.

For each population and source group, report counts, accuracy, the confusion
matrix in poor/good order, poor sensitivity, good specificity, balanced accuracy,
poor AUROC and stepwise average precision, negative log likelihood, Brier score
summed over both classes and top-label ECE with ten equal-width confidence bins.
Report the observed accuracy of the fixed always-good reference and retain
unavailable class metrics as null.
Treat discrimination and threshold behavior separately from calibration.
Use point estimates given the unresolved patient grouping. The quality
endpoint alone supports no claims about DR grading or triage utility.

Independently reconstruct the label join, candidate and exclusion counts,
and all aggregate metrics from the local evidence. Replay the first two
eligible members in archive-name order from each nonempty source group.
Publish aggregate results and an audit receipt. Store image identifiers,
label rows and logits in ignored local artifacts.
