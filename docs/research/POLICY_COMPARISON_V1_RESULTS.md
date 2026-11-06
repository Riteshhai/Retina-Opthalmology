# Independently tuned guarded policies — executed 22 October 2026

**VERIFIED COMPLETE:** the declared development policy comparison and independent
audit. The experiment does not establish a benefit from independent policy
tuning. The production ordinal gate still requires review for all 168 images;
useful validated automatic triage remains unproven. No deployed classifier,
temperature, sidecar, guard, default price or action semantics changed.

The [protocol](POLICY_COMPARISON_V1_PROTOCOL.md) was committed at `4167118` before
execution. The executed source is `c2f8fbb50622631c6001092f7b1da008498f421b`.
The [complete aggregate report](../../reports/policy_comparison_v1/evaluation.json)
retains all scenarios, tuning candidates, fold failures and planned comparisons.
Its SHA256 is
`4dfd99d535d7f7413596caab5ca1fa47629a62126a488b907563656bf5834085`.
The [independent audit](../../reports/policy_comparison_v1/independent_audit.json)
passed. Individual actions, patient identifiers and fold assignments remain local
under ignored artifacts.

The [compact dashboard artifact](../../reports/policy_comparison_v1/dashboard_summary.json)
is derived by `scripts/summarize_policy_comparison.py` and binds the full measured
report's exact SHA256. The [feasibility/coverage figure](../../reports/policy_comparison_v1/scenario_statuses.png)
distinguishes incomplete outcomes (dash) from observed zero automatic coverage.
The API and experiment explorer read this derived artifact; the complete measured
candidate report remains preserved without modification.

## Population and comparison design

The study reuses bound first-pass scores for the 168 DeepDRiD selection images /
42 patients, including 74 referable and 50 severe image labels. All original image
bytes, manifest order, cache binding and selected model/calibration/sidecar hashes
were verified. No new classifier inference, calibration fitting, confirmation,
official test or external evaluation was performed.

Three fixed patient folds each exclude 14 patients / 56 images for evaluation and
use the other 28 patients / 112 images for policy tuning. No patient appears on
both sides of a fold. The classifier was already selected using all 42 patients,
and this population has been inspected in earlier experiments. Consequently this
is development policy cross-fitting, not independent end-to-end validation.

Each of eight guarded methods is evaluated in three separately reported modes:
the unchanged deployed deterministic patient-max ordinal gate (`ordinal`), the
existing conventional LAC gate (`lac`), and an explicitly labeled no-conformal
ablation (`no_cp`). LAC and no-CP are diagnostic modes, not deployment candidates
selected by this experiment. All methods retain the applicable mandatory risk,
severity, quality and entropy guards; these are guarded wrappers, not pure
unrestricted literature baselines.

The four fixed review/recapture budget pairs are (.30,.20), (.60,.40), (.90,.40)
and (1.00,.40). Each method/mode/budget/fold independently chooses an allocation
review price from .05, .10, .25, .50, 1.00 and 2.00 using its tuning cohort. The
untuned reference uses allocation price .25. Recapture price and all other
thresholds/weights remain fixed. Labels enter tuning and outcome evaluation;
allocation itself uses predictions and configured guards, not true labels.

Every candidate and held-fold result is scored with the **same fixed ledger**:

`[severity-weighted referable automatic misses + severity-weighted severe
recaptures + .25 × reviews + .20 × recaptures] / number of images`,

using grade weights (0,0,3,6,10). Changing the allocation review price does not
change the evaluation review cost. The ledger is an unvalidated research proxy,
not estimated clinical utility. A severe recapture is a potentially delayed
deferral, reported separately from an automatic referable miss. Deferral is a
recommendation, not evidence that care was completed.

## Complete accounting, including failures

All 192 planned scenarios are present: 96 independently tuned scenarios and 96
default references. Only 78 have actions for all three evaluation folds; the other
114 retain **null pooled metrics**, rather than a favorable zero or an average
over the folds that happened to fit. Of 1,728 tuning candidates, 786 evaluated
and 942 overloaded. All six prices were retained for every tuning task.

| Mode | Fully evaluated scenarios | Incomplete scenarios | Total |
| --- | ---: | ---: | ---: |
| Deployed ordinal | 16 | 48 | 64 |
| LAC diagnostic | 16 | 48 | 64 |
| No-CP ablation | 46 | 18 | 64 |
| All modes | 78 | 114 | 192 |

The 576 held-fold records comprise 256 evaluated allocations, 163 actual
held-fold overloads and 157 skipped evaluations because tuning was untunable.
An **untunable** fold means all six prices overloaded on its 112-image tuning
cohort. Its retained diagnostics therefore describe tuning N=112; the 56-image
evaluation allocation was not attempted. An **evaluation overload** instead
describes an attempted allocation on the excluded 56 images. These failure
types must not be combined into a claim that every skipped held cohort was
itself tested and overloaded.

Each capacity is an independent floor within its 56-image evaluation batch.
Pooled capacities sum those applied floors: the four review/recapture capacity
pairs are (48,33), (99,66), (150,66) and (168,66). They are not recomputed by
applying a new floor to all 168 images. Every fully evaluated allocation respects
both actual capacities. Capacity maxima do not require a method to use all slots.

## Dual-budget outcomes

The table retains every dual-budget mode/budget with three evaluable folds for
both tuned and default variants. Counts are pooled over all 168 images / 42
patients. `Auto` means provisional `AUTO_NONREFERABLE`, not correct diagnosis.
The displayed proxy uses the fixed ledger above; lower is cheaper under that
ledger only. Every row below has zero severe automatic misses.

| Mode | Review/recapture budget | Variant | Auto | Review | Recapture | Referable automatic misses | Severe recaptures | Proxy / image |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinal | 1.00/.40 | Tuned | 0 | 168 | 0 | 0 | 0 | .250000 |
| Ordinal | 1.00/.40 | Default | 0 | 168 | 0 | 0 | 0 | .250000 |
| LAC diagnostic | 1.00/.40 | Tuned | 2 | 159 | 7 | 0 | 2 | .316369 |
| LAC diagnostic | 1.00/.40 | Default | 0 | 168 | 0 | 0 | 0 | .250000 |
| No-CP ablation | .60/.40 | Tuned | 18 | 98 | 52 | 1 | 5 | .404167 |
| No-CP ablation | .60/.40 | Default | 18 | 99 | 51 | 1 | 5 | .404464 |
| No-CP ablation | .90/.40 | Tuned | 3 | 145 | 20 | 0 | 2 | .311012 |
| No-CP ablation | .90/.40 | Default | 2 | 150 | 16 | 0 | 1 | .277976 |
| No-CP ablation | 1.00/.40 | Tuned | 3 | 151 | 14 | 0 | 2 | .312798 |
| No-CP ablation | 1.00/.40 | Default | 0 | 168 | 0 | 0 | 0 | .250000 |

The deployed ordinal gate permits no automatic cases under any price. Below a
100% review budget, all three tuned dual-budget folds are untunable and all three
default folds overload. At the 100% review budget, every method/variant reviews
all cases. Zero misses in this situation provide no evidence of useful automatic
triage or a ranking advantage.

LAC dual-budget allocations are incomplete at .30/.20 and .60/.40 for both
variants. At .90/.40, tuned fold statuses are untunable, evaluation overload and
evaluated; the default has two evaluated folds and one overload. Its pooled
outcomes remain null. No-CP dual-budget allocations are incomplete at .30/.20.
These scenarios remain visible in the report rather than being omitted from
the experiment denominator.

Chosen dual-budget allocation prices for folds 0/1/2 were .05/.05/.05 for ordinal
1.00/.40; .50/1.00/.50 for LAC 1.00/.40; 2.00/2.00/.05 for no-CP .60/.40;
.05/1.00/.05 for no-CP .90/.40; and .50/1.00/.50 for no-CP 1.00/.40.
These are fitted decision parameters, not revised evaluation costs or a
recommendation for deployment.

Independent tuning can worsen excluded-fold outcomes. At the 100% review budget,
LAC tuning introduces two severe recaptures and raises proxy cost from .250000
to .316369 while clearing two images (1.19%). No-CP tuning at .90/.40 and 1.00/.40
also raises proxy cost and has two severe recaptures. The largest displayed
automatic coverage is 18/168 (10.71%) in the no-CP .60/.40 ablation, accompanied by
one referable automatic miss and five severe recaptures in both variants. The
tiny proxy decrease there exchanges one review for one recapture. These are
unequal actual workloads; none establishes a resource-matched policy advantage.

## Planned paired comparisons and uncertainty

All 180 planned comparisons are retained: tuned dual-budget versus seven tuned
guarded comparators, and each tuned method versus its own default, at every
mode/budget. Only 73 pairs have complete outcomes on both sides. Of those,
19 match **both actual review and recapture counts within every excluded fold**;
54 have unequal workloads. The remaining 107 pairs are not estimable and have
no numeric differences. Equal budget fractions alone are not workload matching.

| Mode | Planned pairs | Estimable pairs | Actual workloads match in every fold |
| --- | ---: | ---: | ---: |
| Ordinal | 60 | 15 | 15 |
| LAC diagnostic | 60 | 15 | 2 |
| No-CP ablation | 60 | 43 | 2 |
| All modes | 180 | 73 | 19 |

All 19 matched pairs have identical individual actions; proxy cost, automatic
indicator and referable-auto miss differences are zero, with [0,0] descriptive
intervals. Fifteen matched pairs are ordinal all-review cases. Of the other
four, LAC single-budget tuned/default is also all review. Only three matched
pairs have any automatic cases: LAC dual-budget versus naive at 1.00/.40 has two
each; no-CP entropy-only tuned/default at .90/.40 has ten each; and no-CP
single-budget tuned/default at .90/.40 has eighteen each. None has a matched
workload benefit. All nonzero reported paired differences occur with unequal
actual workloads and cannot establish ranking superiority at matched resources.

For each estimable pair, the report gives 95% paired patient-cluster percentile
intervals from 1,000 resamples, seed 20261022, for fixed per-image proxy cost,
automatic indicator and referable-auto miss indicator. For example, tuned minus
default dual-budget proxy differences are +.066369 (LAC 1.00/.40; interval
[-.009524,.211905]), -.000298 (no-CP .60/.40; [-.000893,.000000]), +.033036
(no-CP .90/.40; [-.005357,.105960]) and +.062798 (no-CP 1.00/.40;
[-.013393,.208951]). All four comparisons have unequal actual workloads.

Resampling retains each patient's fixed actions on both sides; prices are not
refitted and the allocator is not rerun. The intervals condition on overlapping
cross-fit tuning populations and fixed actions, use no multiplicity adjustment,
and do not guarantee capacity compliance in a resampled cohort. They are
descriptive development uncertainty, not prospective safety or full training and
policy-selection uncertainty. Small observed miss counts do not establish safety.

## Verification and interpretation

The independent audit reconstructed patient folds, all 1,728 candidate outcomes,
all 131 selected prices where tuning succeeded, all 576 held-fold records, all
192 pooled/null outcomes and all 180 comparisons. It independently reconstructed
all 219 reported bootstrap intervals. It checked exact source/protocol bindings,
all 168 original image hashes, selected checkpoint/calibration/conformal artifacts,
fold separation, declared tie rules, fixed ledgers, guard/capacity restrictions
and absence of patient identifiers/actions from the public report. Historical
triage and prior confirmation/development/pilot reports remained unchanged.

Allocator replay uses the production linear-program solver; independent audit
mathematics checks guards, counts, ledger, candidate ranks, capacities and
bootstrap rather than introducing a second LP implementation. Equal-cost LP
solutions can depend on solver version and row order; the recorded NumPy 2.4.6,
SciPy 1.17.1 and pandas 3.0.5 identify this execution, without a cross-version
assignment-identity claim.

The experiment closes the independently tuned comparator execution gap while
retaining a negative operational result: guards and uncertainty dominate this
small development population, production ordinal automatic coverage remains
zero, and no matched-workload advantage is demonstrated. All modes, budgets,
methods and tuning outcomes are retained; no winner or parameter is promoted.
Useful triage, official APTOS reproduction and independent external transport
remain separate unfinished research requirements. Formal manuscript drafting
remains deferred.
