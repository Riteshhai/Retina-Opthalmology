# Direct referability prediction sets for guarded development triage

The experiment compares three conformal gates on 168 reused DeepDRiD
selection images from 42 patients. Each gate uses 84 conformal-calibration
images from 21 separate patients and the stored first-pass temperature
fitted on another 21 patients. The diagnostic fits score thresholds on
saved probabilities. It uses the original risk, quality and entropy
guards and the fixed dual-budget allocator.
[Locked protocol](REFERABILITY_GATE_V1_PROTOCOL.md).

## Prediction sets and pointwise routing

The sets describe inclusion of the true referability label. The pointwise
recommendations omit cohort capacity limits.

| Alpha | Method | Image inclusion | Patient all-views inclusion | Mean binary set size | Non-referable singleton sets | Pointwise auto / recapture / review |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| .10 | Projected ordinal | 168/168 | 42/42 | 2.000 | 0 | 0 / 0 / 168 |
| .10 | Projected LAC | 163/168 | 39/42 | 1.732 | 21 | 6 / 15 / 147 |
| .10 | Binary LAC | 163/168 | 39/42 | 1.607 | 34 | 9 / 25 / 134 |
| .05 | Projected ordinal | 168/168 | 42/42 | 2.000 | 0 | 0 / 0 / 168 |
| .05 | Projected LAC | 167/168 | 41/42 | 1.804 | 16 | 6 / 10 / 152 |
| .05 | Binary LAC | 163/168 | 39/42 | 1.661 | 29 | 6 / 23 / 139 |

Binary LAC yielded the smallest mean sets at both alphas. At alpha .05,
its observed patient all-views inclusion was 39/42 versus 41/42 for
projected LAC. Binary LAC excluded the true referable label on 2 of 40
grade-3 images at each alpha. Both images were recommended for recapture
under pointwise routing. This shows why set inclusion, pointwise routing
and budgeted outcomes need separate accounting. No raw prediction sets
were empty.

## Dual-budget allocations

Every method overloaded at the .30/.20 and .60/.40 review/recapture
budget pairs. All six settings allocated every image to specialist review
at 1.00/.40 under the fixed decision costs. The remaining pair had a
review capacity of 151 images and a recapture capacity of 67.

| Alpha | Method at .90/.40 | Mandatory review | Auto / recapture / review | Referable auto misses | Severe recaptures | Observed proxy per image |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| .10 | Projected ordinal | 168 | Overloaded | — | — | — |
| .10 | Projected LAC | 147 | 5 / 12 / 151 | 0 | 0 | .2390 |
| .10 | Binary LAC | 134 | 3 / 14 / 151 | 0 | 0 | .2414 |
| .05 | Projected ordinal | 168 | Overloaded | — | — | — |
| .05 | Projected LAC | 152 | Overloaded | — | — | — |
| .05 | Binary LAC | 139 | 3 / 14 / 151 | 0 | 0 | .2414 |

Direct calibration reduced mandatory review and allowed a feasible
allocation at alpha .05 where both projected methods overloaded. At
alpha .10, it led to two fewer automatic recommendations than projected
LAC under the same budgets. With the fixed prediction-based costs, a
narrower set creates options for the allocator and does not determine
which eligible images it clears. Across 24 scenarios, nine were feasible
and 15 overloaded.

Automatic clearance occurred on at most 5 of 168 images in any budgeted
scenario. Zero referable misses in these small automatic subsets provide
limited evidence about risk. All results are descriptive and use the
same development patients who informed model choice. The study establishes
neither clinical safety nor independent validation.
[Complete aggregate evaluation](../../reports/referability_gate_v1/evaluation.json).

The independent audit rehashed 336 source photographs, reconstructed
six threshold fits, 1,008 binary sets and 1,008 pointwise actions, and
verified all 24 allocations or overloads. The largest difference from
the independently computed minimum allocation cost was 1.42 × 10⁻¹⁴.
[Audit receipt](../../reports/referability_gate_v1/independent_audit.json).
