# Patient outcomes under image-based referability gates

The saved allocation with the most automatic recommendations selected five
of 168 photographs. Those photographs belonged to two patients, and only one
of 42 patients had automatic recommendations for all four views. The three
settings that fit the 90% image review cap needed more than the corresponding
37 patient review slots.

This follow-up covers six pointwise fits and 24 saved scenarios. Each of the
42 patients had two views per eye. The worst eye grade agreed with the recorded
patient grade in every case. Twenty-one patients had at least one referable eye.
[Locked protocol](PATIENT_TRIAGE_V1_PROTOCOL.md) ·
[Original development study](REFERABILITY_GATE_V1_RESULTS.md).

## Patient outcomes from saved image recommendations

Any review recommendation takes priority. A patient with recapture and no
review is counted as recapture. Full automatic clearance requires four
automatic image recommendations. This summary gives each patient one action
for counting purposes. In practice, a patient with a review recommendation
may also need a photograph recaptured.

| Alpha | Gate | Pointwise automatic photos | Patients with any automatic photo | Patient automatic / recapture / review |
| ---: | --- | ---: | ---: | ---: |
| 0.10 | Projected ordinal | 0 | 0 | 0 / 0 / 42 |
| 0.10 | Projected LAC | 6 | 2 | 1 / 1 / 40 |
| 0.10 | Binary LAC | 9 | 4 | 1 / 2 / 39 |
| 0.05 | Projected ordinal | 0 | 0 | 0 / 0 / 42 |
| 0.05 | Projected LAC | 6 | 2 | 1 / 0 / 41 |
| 0.05 | Binary LAC | 6 | 2 | 1 / 2 / 39 |

The nine pointwise automatic images for binary LAC at .10 reached four
patients. Three of those patients still had deferred views. Across the
evaluated image budgets, only projected LAC at .10 produced one fully
automatic patient. That patient had no referable eye under the dataset
labels. The observed full clearance miss fraction was 0/1 for this
allocation and undefined for the other eight evaluated allocations.

## Capacity counted in patients

| Review / recapture budget | Patient review / recapture caps | Feasible image settings | Feasible patient settings |
| --- | ---: | ---: | ---: |
| 30% / 20% | 12 / 8 | 0/6 | 0/6 |
| 60% / 40% | 25 / 16 | 0/6 | 0/6 |
| 90% / 40% | 37 / 16 | 3/6 | 0/6 |
| 100% / 40% | 42 / 16 | 6/6 | 6/6 |

At 90% review capacity, each of the three evaluated image allocations
rolled up to 41 patients with review recommendations. The minimums in the
table apply to the fixed patient guards and allow recapture-only patients
to use available recapture slots. They can differ from the observed rollups
because the image allocation can also assign optional reviews.

| Alpha | Gate at 90% / 40% | Automatic photos | Observed patient automatic / recapture / review | Minimum patient reviews | Shortfall versus 37 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.10 | Projected LAC | 5 | 1 / 0 / 41 | 40 | 3 |
| 0.10 | Binary LAC | 3 | 0 / 1 / 41 | 39 | 2 |
| 0.05 | Binary LAC | 3 | 0 / 1 / 41 | 39 | 2 |

All six patient scenarios were feasible at the 100% review cap, where the
saved image allocator recommended review for all 42 patients under each
setting. The separate patient feasibility calculation found at most one
patient eligible for full automatic clearance in any feasible setting.
It leaves the individual capacity assignments unspecified.

## Prospective planning

| Illustrative upper miss rate | Fully automatic patients with zero misses | Fully automatic patients with one miss |
| ---: | ---: | ---: |
| 5% | 59 | 93 |
| 2% | 149 | 236 |
| 1% | 299 | 473 |

These are the minimum fully automatic denominators for a one-sided exact
95% binomial upper bound at or below the chosen value. The calculation
assumes independent outcomes under one fixed rule and requires reference
labels for the automatically cleared patients. The development analysis
reused model-selection patients and coupled image decisions through shared
capacity. The sample sizes apply to a future independent assessment.
[Clopper and Pearson](https://doi.org/10.1093/biomet/26.4.404).

The independent audit checked the 168 manifest rows, all 15 observed action
rollups, all 24 patient-capacity scenarios and all six binomial minima.
[Aggregate evaluation](../../reports/patient_triage_v1/evaluation.json) ·
[Audit receipt](../../reports/patient_triage_v1/independent_audit.json).

## Reproduce

Run both scripts from a clean checkout of
`3c9ccd47384cd8367bcf16dba29789dac791c5ee` with the bound local development
manifest and ordinal and referability witnesses.

```text
python scripts/run_patient_triage.py --source-commit 3c9ccd47384cd8367bcf16dba29789dac791c5ee
python scripts/audit_patient_triage.py
```
