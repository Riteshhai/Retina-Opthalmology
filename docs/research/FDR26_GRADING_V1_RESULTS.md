# Frozen DeepDRiD grading on FDR26 v3

On 2,849 eligible FDR26 photos, the frozen DeepDRiD classifier matched
794 of the five-grade ALFEDIAM labels (27.87%). Macro-F1 was 0.268 and
quadratic-weighted kappa was 0.381. Predictions were at least two grade
codes below the reference on 804 photos (28.22%).
[Locked protocol](FDR26_GRADING_V1_PROTOCOL.md).

The model used the selected 224-pixel MobileNetV3-Small checkpoint and
its existing first-pass temperature of 1.4621494532. Each photo received
one CPU forward. The three sensitivity views reuse those predictions.
[Data readiness audit](FDR26_READINESS_V1_RESULTS.md).

| Population | Photos | Grade accuracy | Macro-F1 | Quadratic kappa | Grade ≥2 AUROC | Grade ≥2 sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 2,849 | 27.87% | 0.268 | 0.381 | 0.703 | 49.06% |
| Fingerprint representatives | 2,742 | 27.94% | 0.268 | 0.381 | 0.712 | 48.78% |
| Extended prefix exclusions | 2,761 | 27.78% | 0.266 | 0.387 | 0.708 | 49.36% |
| Extended prefix representatives | 2,673 | 27.87% | 0.267 | 0.385 | 0.717 | 49.13% |

## Errors by grade

| Reference grade | Photos | Exact matches | Recall |
| ---: | ---: | ---: | ---: |
| 0 | 490 | 313 | 63.88% |
| 1 | 659 | 128 | 19.42% |
| 2 | 522 | 58 | 11.11% |
| 3 | 606 | 141 | 23.27% |
| 4 | 572 | 154 | 26.92% |

The calibrated grade ≥2 score had AUROC 0.703 and average precision
0.757. At the fixed .50 threshold it detected 834 of 1,700 positive
photos and missed 866. It correctly rejected 924 of 1,149 negative
photos. Binary accuracy was 61.71%, compared with 59.67% for always
predicting grade ≥2. These are image-level results against source
grade codes. The grade-enriched dataset and different lesion criteria
limit interpretation across cohorts. The unconfirmed filename groups
were used only to extend exclusions, and the workbook supplies no
clinical referral, quality or macular edema endpoint.

The frozen temperature reduced NLL from 2.057 to 1.738 and confidence
ECE from 0.352 to 0.251. It preserved all five-way grade decisions.
Summing three softmax probabilities changed the binary score and its
threshold decisions. Sensitivity rose from 46.76% to 49.06% while
specificity moved from 80.50% to 80.42%.

The independent audit reproduced eight aggregate metric views and
verified all 2,849 probability records. It replayed ten photos, two
per grade, and matched their saved logits exactly. The first audit
stopped because a float32 softmax sum differed from 1 by 2.012e-7.
The revised audit accepts normalization error up to 1e-6 and retains
its stricter 2e-7 elementwise probability comparison.
[Aggregate report](../../reports/fdr26_grading_v1/evaluation.json) ·
[Independent audit](../../reports/fdr26_grading_v1/independent_audit.json).

## Reproduce

Reproduce scores from clean commit `af1e3362cfd4db6b70537c9d2602d263efaa4b86`
using the verified local archive, readiness witness, selected DR checkpoint
and DeepDRiD calibration.

```text
python scripts/evaluate_fdr26_grading.py --source-commit af1e3362cfd4db6b70537c9d2602d263efaa4b86
```

Run `python scripts/audit_fdr26_grading.py` from audit commit
`41952cdc45c9361bc30c204ca35899595bb24ace` or a later checkout with the same
audit source. FDR26 v3 is attributed to Sanae Lamti, Noureddine Kerzazi,
Rachid Sekhsoukh and Asmae Guechati.
[Author deposit](https://doi.org/10.5281/zenodo.23101200) ·
[Data article](https://doi.org/10.1016/j.dib.2026.113379) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
