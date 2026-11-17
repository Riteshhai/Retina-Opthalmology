# DeepDRiD paired-view acquisition

At eight extra views, the estimated-value ranking reduced missed referable
eyes from 8 to 6 and the mean research ledger from 0.5833 to 0.4536 per eye.
It had the lowest observed ledger among the five rankings at that workload.
All 16 matched-workload cost intervals included zero. The rule that acquired
only above the estimated cost used four extra views at every interior cap,
corrected no missed referrals, and raised the mean ledger to 0.5929.
[Locked protocol](PAIRED_VIEW_V1_PROTOCOL.md).

## Population and fit

The frozen first-pass pipeline scored 1,016 photos. Ridge regression learned
signed acquisition benefit from 424 eyes in 212 training patients. Evaluation
used 84 eyes from 42 separate selection patients, including 37 referable and
25 severe eyes under the study definitions. The frozen classifiers previously
used these training and selection partitions. The evidence describes a
patient-disjoint value-model evaluation within development.

The five ridge predictors came from the initial view alone. The mean gross
training gain was 0.00295 ledger units per eye, before the .20 acquisition
charge. Observed two-view benefit exceeded that charge for 16 of 424 training
eyes. At evaluation time, the policy ranked the initial-view scores before
applying the fixed mean of the two probability vectors on selected pairs.

## Workload and outcomes

| Policy | Extra views | Eye grade accuracy | Eye missed referrals (all / severe) | Patient missed referrals (all / severe) | Mean ledger per eye |
| --- | ---: | ---: | ---: | ---: | ---: |
| First view only | 0 | 63.1% | 8 / 4 | 3 / 1 | 0.5833 |
| Value ranking at 10% | 8 | 63.1% | 6 / 3 | 2 / 1 | 0.4536 |
| Value ranking at 25% | 21 | 61.9% | 6 / 3 | 2 / 1 | 0.4875 |
| Value ranking at 50% | 42 | 63.1% | 6 / 3 | 2 / 1 | 0.5375 |
| Value ranking at 75% | 63 | 59.5% | 6 / 3 | 2 / 1 | 0.5875 |
| Both recorded views | 84 | 61.9% | 6 / 3 | 2 / 1 | 0.6286 |
| Acquire when estimated value exceeds cost | 4 | 63.1% | 8 / 4 | 3 / 1 | 0.5929 |

The review proxy counts rose from 36 at first view to 38 at eight extra views.
This workload is separate from the acquisition budget. The severity-weighted
miss term uses released eye grades. A severe missed referral means a true
grade 3 or 4 predicted below grade 2. Patient misses use the worse eye.
The released fields support a retrospective comparison. Capture chronology
and the result of requesting a future retake remain unobserved.
[DeepDRiD data paper](https://doi.org/10.1016/j.patter.2022.100512).

## Exact-workload comparisons

| Extra views | Value minus quality | Value minus entropy | Value minus referable risk | Value minus seeded random |
| ---: | ---: | ---: | ---: | ---: |
| 8 | −0.1161 | −0.1161 | −0.1488 | −0.0327 |
| 21 | −0.1161 | +0.0060 | −0.1458 | −0.0327 |
| 42 | −0.1161 | +0.0089 | −0.1071 | −0.0268 |
| 63 | 0.0000 | +0.0089 | +0.0089 | +0.0060 |

Differences are mean research ledger units per eye. At eight views the
value-versus-entropy difference was −0.1161, with a 95% paired patient
interval of [−0.3530, +0.0095]. Entropy had a lower observed ledger at
21, 42, and 63 extra views. These intervals hold acquisition assignments fixed
and resample patients as units. The random comparator is one seeded
ordering.

## Audit and reproduction

The independent auditor reconstructed the ridge fit, first-view rankings,
26 acquisition scenarios, and 16 exact-workload comparisons from the frozen
roster and saved scores. Keep the private manifest and score witness local.
To reproduce, use a clean checkout of `6017d0c889d539b9d8c74ab33907015388040acf`
with empty study-output paths and the original bound manifest and artifacts.

```text
python -u scripts/run_paired_view_v1.py --source-commit 6017d0c889d539b9d8c74ab33907015388040acf
python scripts/audit_paired_view_v1.py
python scripts/plot_paired_view_v1.py
```

Select `paired_view_v1` in **Experiment Explorer** to choose an audited
acquisition budget and compare workloads and outcomes.
[Aggregate report](../../reports/paired_view_v1/evaluation.json) ·
[Audit receipt](../../reports/paired_view_v1/independent_audit.json) ·
[Acquisition workload figure](../../reports/paired_view_v1/acquisition_workload.png).
