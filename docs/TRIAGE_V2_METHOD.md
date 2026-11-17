# Prospective v2 non-referable triage

This implementation follows MASTER_SPEC v1.0. It lives in `qsart/triage_v2.py`;
the historical `qsart/triage.py` and its AUTO_SCREEN reports remain unchanged.
The new module has software verification using synthetic scores. Its defaults
have not been tuned or validated as a clinical policy. Existing v1 policy results
must not be relabeled as results for this method.

## Decision semantics and guards

AUTO_NONREFERABLE is a provisional automated non-referable screening outcome.
It is eligible only when all of these conditions hold:

- The predicted grade is below the configured referable grade (default 2).
- The summed referable probability is below `review_risk` (default 0.50).
- The summed probability at or above `severe_grade` (default 3) is below
  `severe_risk` (default 0.25).
- Good-quality probability is at least `minimum_quality` (default 0.50).
- Normalized predictive entropy is below `uncertainty_threshold` (default 0.75).
- If a conformal set is supplied, it is nonempty and contains only grades below
  the configured referable grade.

Predicted referable grade, high referable probability, high severe probability,
an empty conformal set, or a set containing any referable grade requires review,
including when quality is poor. High entropy with adequate quality also requires
review. Other poor-quality cases require recapture or review; review can absorb
recapture overflow only if spare review capacity exists. Missing conformal output
does not imply a successful conformal safety check or a coverage guarantee.

Pointwise decisions have no population capacity guarantee. Batch allocation
enforces the guards and both independent integer capacities. For N images,
review capacity is floor(B_review × N), and recapture capacity is independently
floor(B_recapture × N). Decimal representations of the declared fractions avoid
adding epsilon slots. A zero-capacity cohort succeeds only if every image is
eligible for clearance.

For M mandatory reviews and Q remaining poor-quality cases, feasibility requires
M ≤ review capacity and M + Q ≤ review capacity + recapture capacity. Under the
review-only baseline, all Q also join M. An infeasible cohort raises the typed
`BudgetOverloaded` exception with safe aggregate counts, capacities, shortfalls
and `unsafe_clearance_permitted=false`. It produces no forced-clearance result.

## Prediction-only research objective

Allocation minimizes the sum of per-image action proxy costs using a linear
program with one assignment per image. This transportation constraint structure
has integral vertices; the implementation checks integral assignments and audits
the caps and guards after solving. Labels, patient identifiers and actual future
care outcomes never enter allocation.

The expected severity-weighted missed-referral score is

`s = Σ[g ≥ referable_grade] p(g) × w(g)`.

Default grade weights are (0, 0, 3, 6, 10); they are configurable research choices,
not measured clinical costs. Specialist review costs 0.25. Recapture costs
`0.20 + s + 0.5 × good_quality_probability`, retaining expected disease risk
instead of assuming that recapture cures it. Recapture is allowed only for
poor-quality images, and high-risk guards still require review.

All eight deployed strategy wrappers share the mandatory guards. They differ in
the clearance-cost score used for additional optional review:

| Strategy | Clearance proxy | Extra restriction |
|---|---|---|
| plain | 0 | Eligible images stay automatic; required deferrals remain |
| confidence_only | 1 − maximum grade probability | None |
| entropy_only | normalized entropy | None |
| severity_only | s | None |
| quality_only | 1 − good-quality probability | None |
| single_budget | s + quality weight × (1−q) + entropy weight × u | Review only; poor-quality cases also require review |
| naive | (1−q) + u | None |
| dual_budget | s + quality weight × (1−q) + entropy weight × u | Independent review and recapture capacities |

Default quality and entropy weights are 1 and 0.5. Unused capacity is allowed;
capacity is a maximum, not an instruction to fill every slot. Common guards mean
these wrappers are guarded deployed comparators, not unrestricted reproductions
of raw published baselines. Independently tuned development comparisons and
matched actual-workload analysis remain separate research work.

`no_deferral_predictions` is the distinct raw classifier baseline. It returns
five-grade and binary referability predictions with no action field. In
particular, a predicted referable result is never presented as non-referable
clearance merely because the raw classifier does not defer.

## Evaluation denominators

`screening_metrics` consumes true grades only after allocation. Every true
referable case assigned AUTO_NONREFERABLE counts as a missed referable case,
irrespective of its five-grade prediction. It reports misses per 1,000 of all
images, the missed fraction among all true referable images, and false omission
among automatic clearances separately. It also reports severe cases sent to
recapture, their fraction among all true severe cases, review/recapture counts
and rates, both integer capacities, and explicit proxy cost components. Empty
conditional denominators produce null rather than a favorable zero.

The unit is an image. Patient-level referral, completed specialist decisions,
recapture success, prospective safety and external transport are not established
by these software outputs. Observed severe recaptures can still occur from model
false negatives despite the prediction-based severe-risk guard.

## Interfaces

- `pointwise_action(probabilities, quality, uncertainty, config, conformal_set=None)`
  returns `(Action, reason_codes)`.
- `allocate(probabilities, quality, uncertainty, config, method, conformal_sets=None)`
  returns one Action per image or raises BudgetOverloaded.
- `allocate_rows(rows, config, method)` consumes current engine output dictionaries
  (`grade_probabilities`, `quality.poor_probability`, `uncertainty.entropy`) and
  optional top-level `conformal_set`. Extra fields such as true grades are ignored.
- `screening_metrics(grades, probabilities, actions, config)` is evaluation only.

The v2 policy and conformal artifacts must be frozen before a new independent
evaluation. The consumed 340-image v1 confirmation cohort can support explicitly
retrospective software/integration evidence, not a fresh confirmatory claim.
