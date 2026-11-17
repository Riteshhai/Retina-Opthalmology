# Horizontal-flip inference at matched workloads

This study asks whether first-pass scores direct extra inference to images
whose predictions benefit. It follows the earlier CPU timing benchmark and
uses the full frozen DeepDRiD development selection cohort. Count corrections
and newly introduced errors, as both can occur after test-time augmentation.
[Prior CPU workflow](../../reports/spec_v1/end_to_end_cpu.json) ·
[Shanmugam et al. on test-time augmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html).

## Fixed evidence and workloads

Bind the 168 selection rows from 42 patients in their original manifest order,
the cached single-pass, all-flip and 25% budget profiles, the originating
report, and the matching classifier, quality model and calibration artifacts.
The original conformal sidecar supplies the saved sets for each probability
path. Verify the cohort labels and patient order against the ordinal score
witness. All patients were previously used for development selection.

Treat the cached first pass and horizontal-flip logit average as paired
predictions. At each budget use the original 21 requests of eight images.
The 25% and 50% budgets assign exactly two and four extra passes per request.
Rank all images in each request by descending first-pass score, with ties
broken by manifest order. Compare normalized entropy, referable probability,
and the existing controller score, the mean of those two values. These are
fixed rankings with equal slot counts. Verify the 25% combined controller
against the saved mixed profile, excluding timing fields. Include the
no-flip and all-flip profiles as workload anchors.

## Outcomes

The primary endpoint is the change in the number of correctly classified
five-grade images relative to first pass. A second endpoint uses grade 2
or higher as the referable reference and the saved referable probability
at least .50 as the binary classifier threshold. For both endpoints
count first-pass errors corrected and first-pass correct predictions lost.
Record false negatives and false positives for referability. Give grade
denominators, overall quality accuracy, and patient counts with a grade
correction or a new grade error. Report worst-eye patient referability
by aggregating the image-level binary flags with an any-view rule.

For the deterministic rankings and workload anchors, also record image and
patient all-views inclusion and mean set size for the saved .10 LAC and
ordinal sets. Replay the deployed ordinal guard using the saved probabilities,
quality and entropy for each selected path. Report its pointwise review,
recapture and provisional automatic counts, plus referable automatic misses.
The original images and labels provide no observed care outcomes.

## Exact random comparison and retrospective bounds

At each budget, let uniform random selection choose that request's fixed
number of slots without replacement. Combine the request distributions
independently. Compute the exact distribution of the change in correct
image counts by enumerating each request's subsets and convolving their
integer counts. Report the expected change, the 2.5th and 97.5th percentiles
of that fixed-cohort distribution, and the probability of matching or
exceeding each fixed ranking. Compute the minimum and maximum reachable
changes at the same slot count using the observed per-image gains. Each
endpoint gets its own bounds. These label-aware extremes describe the
available opportunity and harm in the saved pairs.

Independently reconstruct the rankings, paired outcome counts, set coverage,
routing and exact random distributions. Replay both probability paths on
the first complete request with the bound model on two CPU threads. Record
score agreement within absolute tolerance 1e-6 with exact decisions and
conformal sets. Keep per-image evidence local. The study measures recorded
outcomes at equal extra-pass counts on reused development patients.
Random-assignment percentiles describe this cohort. Workload counts carry
no new latency or generalization estimate.
