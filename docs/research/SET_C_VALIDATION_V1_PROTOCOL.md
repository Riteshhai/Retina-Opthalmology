# DeepDRiD Set-C held-out patient validation

Lock this protocol and the executable study before scoring. Evaluate the
released regular-fundus Set-C cohort once with the frozen DeepDRiD production
checkpoints and rules. The authors used Set-C as the final online evaluation
split from the same release. [DeepDRiD data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/).

## Population and inference

Use the 396 eligible photos from 99 patients in frozen readiness-witness order.
The readiness audit excluded one entire patient after a cross-split retinal
fingerprint match. It established two distinct photos per eye and identical
released grades across the two photos of each eye. Use the supplied eye grades
and overall quality flags. Require the full original frozen group in each
analysis. Verify the readiness report, independent audit, witness and source
photo bytes against their frozen digests. The shared-release design and overlap
screen limit inference about independence. [Readiness evidence](DEEPDRID_ONLINE_READINESS_V1_RESULTS.md).

Use the selected first-pass MobileNetV3 DR and quality models through
`InferenceEngine`, with frozen production temperatures, small CPU batches and
zero extra passes. The bound first-pass sidecar supplies the independently
temperature-scaled patient-max ordinal and LAC sets at alpha .10. The DR
probabilities used for routing keep the production temperature. Use original
released pixels with production preprocessing. Record the source commit, input
digests and runtime versions. Fit no model, temperature, conformal threshold
or decision rule on Set-C. No APTOS test or previously consumed external cohort
supplies a result in this run.

## Prespecified policies and capacities

The primary analysis uses the deployed v2 dual-budget policy with the ordinal
gate and the default decision prices (.25 review and .20 recapture). Evaluate
all eight previously studied guarded wrappers using the same policy defaults,
in ordinal, LAC and no-conformal modes. LAC and no conformal are diagnostics.
The guarded wrappers share the v2 risks, quality, entropy and eligibility rules.
The classifier-only reference measures grading and threshold errors without
assigning screening actions. Report raw and calibrated model scores separately
from conformal coverage. [Development comparator protocol](POLICY_COMPARISON_V1_PROTOCOL.md).

| Review / recapture | Image caps | Patient caps |
| --- | ---: | ---: |
| 30% / 20% | 118 / 79 | 29 / 19 |
| 60% / 40% | 237 / 158 | 59 / 39 |
| 90% / 40% | 356 / 158 | 89 / 39 |
| 100% / 40% | 396 / 158 | 99 / 39 |

Apply independent floors to the stated image fractions. Record all 96
mode/budget/method scenarios. Use all 396 photos in each allocation. Report
actual review, recapture and automatic counts. Leave action outcomes unavailable
for overloads and retain every shortfall. Retain the separate pointwise result
for each conformal mode. In a patient-capacity feasibility diagnostic, compute
the pointwise mandatory-review count and add any recapture-only patients beyond
the patient recapture cap. This diagnostic counts patients once and stops at
feasibility and the maximum eligible automatic count. The image-budget rollup
records the actual patient workloads and can exceed the patient caps.

## Outcomes and comparisons

The reference grade for each patient is the maximum released eye grade. Grades
2–4 count as referable and grades 3–4 as severe for this study. Patient action
follows the highest-priority photo recommendation, with specialist review
ahead of recapture. Full automatic clearance requires all four photos to be
automatic. Report full patient clearance misses, their severity, all automatic
image misses, partial automatic exposure and severe recapture recommendations.
The primary measured harm is the sum of grade-weighted full patient clearance
misses divided by 99, with weights `(0, 0, 3, 6, 10)`. Also report the fixed
patient-level proxy, which adds grade weight for severe patients assigned
recapture, .25 per review and .20 per recapture. Use null for a conditional
miss fraction when the automatic denominator is empty. Actions represent
recommendations. Review and recapture outcomes were not observed.

Compare each dual-budget result with the seven corresponding guarded wrappers
in the same mode and capacity pair. Require equal realized review and recapture
counts both by image and by patient for the descriptive paired intervals.
Distinguish matched, unmatched and unavailable pairs and identify identical
action vectors. Use percentile intervals from 2,000 paired whole-patient
resamples with seed `20261116`. Keep the action assignments fixed. Report
dual-budget minus comparator for weighted missed-referral harm, the fixed proxy,
automatic fraction, referable full-clearance misses and severe full-clearance
misses per patient. These intervals describe the observed allocation and do not
provide prospective safety bounds for decisions coupled by shared capacity.

Report the primary scenario outcomes even when every policy overloads or sends
everyone to review. Preserve adverse outcomes and undefined risk estimates.
Relate this evaluation to the existing controlled-degradation and external
studies using their published aggregate evidence. Keep all test patient rows,
predictions and actions private. Independently reconstruct the eligibility,
guards, capacities and reported outcomes from the private evidence before
publishing aggregate tables and a workload figure.
