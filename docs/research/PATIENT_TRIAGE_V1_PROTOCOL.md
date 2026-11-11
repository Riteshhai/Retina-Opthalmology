# Patient outcomes under image-based referability gates

The earlier referability study recorded a small number of provisional automatic
recommendations for individual photographs. This follow-up measures how those
recommendations combine across both eyes and checks feasibility when capacity
is counted in patients. Use the six saved pointwise settings and 24 image-budget
scenarios. The image-level outcomes are already known. Fix the patient rules and
planning calculations before computing any patient-level outcomes.
[Original development study](REFERABILITY_GATE_V1_RESULTS.md).

## Cohort and outcome definitions

Use the exact DeepDRiD development selection order from the bound manifest and
the two locked score and action witnesses. Verify the action witness and source
report against the prior independent audit receipt. Check the patient and grade
order against the ordinal score witness. Require 168 verified photographs
from 42 patients, with two views per eye and consistent grades within each eye.
The patient endpoint uses the worst eye grade, with grade 2 or above considered
referable. Verify its agreement with the source patient grade.

Group saved image actions by patient. The patient action is specialist review
when at least one view needs review, recapture when at least one view needs
recapture and none needs review, and provisional automatic clearance when all
four views are automatic. Report patients with any automatic image, patients
with a mix of automatic and deferred images, and patients with full automatic
clearance. Count both kinds of error separately:

- An automatic image miss occurs when a saved automatic image has a referable
  eye grade. A patient may have one or more such images.
- A full patient clearance miss occurs when all images are automatic and at
  least one eye is referable.

A referable patient with automatic views of a non-referable eye can still have
a review recommendation for the other eye. Report that exposure without
counting it as a full clearance. Report the conditional miss fraction among
patients with any automatic image and among fully automatic patients, with null
for empty denominators. The highest-priority recommendation is a patient-level
summary of the image actions. It does not measure performed care or the number
of photographs that would actually be recaptured.

## Patient capacity calculation

For each pointwise gate, apply the same four review/recapture fractions from
the source study to 42 patients using independent integer floors. Each patient
uses at most one slot in this calculation. Keep every mandatory review. A
recapture-only patient may take a recapture slot or spill into review.
The minimum number of required review slots is the mandatory review count plus
any recapture-only patients beyond recapture capacity. A scenario overloads
when that minimum exceeds review capacity. If feasible, the number of patients
eligible for full automatic clearance is an upper bound on the achievable
automatic count. This capacity calculation does not pick an allocation or
reprice the original image actions.

Report the original image-budget outcome and its patient rollup alongside this
patient-capacity result. Retain unavailable outcomes for original image
overloads. For each observed rollup, compare the summarized review and
recapture patient counts to the patient caps and report excesses.

## Prospective evidence planning

For one fixed patient-level rule evaluated in a future independent population,
calculate the minimum number of fully automatic patients whose one-sided
95% exact binomial upper bound would be at most 5%, 2% or 1%, given either zero
or one observed full clearance miss. Invert the lower binomial tail at 0.05.
These percentages are illustrative planning values. Each sample size counts
fully automatic patients with independently ascertained reference labels.
Overall enrollment would also depend on the prospective full clearance rate.
No binomial confidence bound will be attached to the reused development
cohort or to decisions coupled by a shared capacity.
[Clopper and Pearson](https://doi.org/10.1093/biomet/26.4.404).

An independent audit will reconstruct the patient counts and enumerate the
capacity choices directly. It will also verify the sample size minima by
checking the binomial tail at each result and at one fewer patient. Publish
only counts, configuration and provenance. This analysis uses frozen outcomes
and produces no new model scores or observed care outcomes.
