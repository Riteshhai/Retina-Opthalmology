# Paired frozen APTOS and DeepDRiD grading on FDR26 v3

On the same 2,849 photos, the frozen APTOS pipeline matched 734 released
ALFEDIAM grades (25.76%) and DeepDRiD matched 794 (27.87%). APTOS had a higher
calibrated grade ≥2 AUROC (0.823 vs 0.703) and lower sensitivity at the fixed
0.50 threshold (40.12% vs 49.06%). The same directions held in all three
sensitivity populations. [Locked protocol](FDR26_APTOS_V1_PROTOCOL.md).

The APTOS model was the 224-pixel EfficientNet-B0 checkpoint from seed 2026,
epoch 4, originally designated as primary in the IDRiD protocol. It used its
source temperature of 1.7353179819929263. The comparison reused the saved
DeepDRiD MobileNetV3-Small predictions and source temperature of 1.4621494532
in the same photo order. Both pipelines used their own frozen preprocessing.
The earlier DeepDRiD results prompted this exploratory follow-up on the reused
FDR26 cohort. No model, temperature or threshold was fitted on FDR26.
[Prior APTOS selection](IDRID_EXTERNAL_V1_PROTOCOL.md) ·
[Prior DeepDRiD results](FDR26_GRADING_V1_RESULTS.md).

## Five-grade agreement

| Population | Photos | Both correct | APTOS only correct | DeepDRiD only correct | Neither correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 2,849 | 407 | 327 | 387 | 1,728 |
| Fingerprint representatives | 2,742 | 393 | 314 | 373 | 1,662 |
| Extended prefix exclusions | 2,761 | 395 | 324 | 372 | 1,670 |
| Extended prefix representatives | 2,673 | 384 | 311 | 361 | 1,617 |

In the table below, paired values are APTOS / DeepDRiD. A two-grade undercall
is a prediction at least two codes below the released reference grade.

| Population | Accuracy | Macro-F1 | Quadratic-weighted kappa | Two-grade undercalls |
| --- | ---: | ---: | ---: | ---: |
| Primary | 25.76% / 27.87% | 0.195 / 0.268 | 0.380 / 0.381 | 1,202 / 804 |
| Fingerprint representatives | 25.78% / 27.94% | 0.194 / 0.268 | 0.377 / 0.381 | 1,167 / 783 |
| Extended prefix exclusions | 26.04% / 27.78% | 0.197 / 0.266 | 0.382 / 0.387 | 1,160 / 777 |
| Extended prefix representatives | 26.00% / 27.87% | 0.195 / 0.267 | 0.380 / 0.385 | 1,133 / 760 |

APTOS predicted grade 0 for 2,127 primary photos (74.66%). It matched 486/490 grade 0
and 195/572 grade 4 labels, compared with 313/490 and 154/572 for DeepDRiD.
APTOS matched 10/659 grade 1, 32/522 grade 2 and 11/606 grade 3 photos, compared
with 128/659, 58/522 and 141/606 for DeepDRiD.

## Grade ≥2 ranking and fixed-threshold decisions

The binary score sums the calibrated probabilities for grades 2 through 4.
Scores at or above 0.50 are flagged as grade ≥2. Paired values are APTOS / DeepDRiD.

| Population | AUROC | Average precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Primary | 0.823 / 0.703 | 0.891 / 0.757 | 40.12% / 49.06% | 98.52% / 80.42% |
| Fingerprint representatives | 0.826 / 0.712 | 0.894 / 0.769 | 40.09% / 48.78% | 98.82% / 82.06% |
| Extended prefix exclusions | 0.826 / 0.708 | 0.892 / 0.759 | 40.68% / 49.36% | 98.47% / 80.97% |
| Extended prefix representatives | 0.829 / 0.717 | 0.895 / 0.772 | 40.57% / 49.13% | 98.79% / 82.45% |

At the fixed threshold, APTOS detected 682 of 1,700 positive photos, missed
1,018 and flagged 17 of 1,149 negative photos. DeepDRiD detected 834, missed
866 and flagged 225 negative photos. Overall binary accuracy was 63.67% for
APTOS and 61.71% for DeepDRiD on this grade-enriched cohort. Both source temperatures reduced primary
five-way log loss (APTOS 5.966 to 3.595 and DeepDRiD 2.057 to 1.738) and
confidence ECE (APTOS 0.642 to 0.568 and DeepDRiD 0.352 to 0.251). Temperature
scaling preserved the five-grade argmax decisions and changed the ranking and
threshold decisions for the summed grade ≥2 score. The aggregate report
includes both probability paths and both confusion matrices for every population.

These descriptive results measure photo-level agreement with the released
ALFEDIAM grades. Differences in the training grading rubrics and the
grade-enriched FDR26 cohort limit interpretation across datasets. The study
compares complete pipelines, with differences in training, architecture and
preprocessing. Patient linkage and independence remain unverified. The
filename-prefix sensitivity views extend the overlap exclusions using
unconfirmed filename groups. The data provide no clinical referral, quality
or macular edema outcomes. [FDR26 readiness audit](FDR26_READINESS_V1_RESULTS.md).

## Independent verification

Before FDR26 scoring, all ten source APTOS replay predictions matched the
saved argmax decisions. The largest absolute probability difference was
2.78e-7 against a tolerance of 1e-5. The independent FDR26 audit reconstructed
all four population masks, verified 2,849 probability records per model, and
recomputed all 16 population probability views and all 16 paired correctness
counts. It replayed the first 16-photo APTOS batch and exactly matched the
first ten stored logits. [Aggregate report](../../reports/fdr26_aptos_v1/evaluation.json) ·
[Independent audit](../../reports/fdr26_aptos_v1/independent_audit.json).

Use clean source commit `799f70874b0fab00284f1dcb31d9e4fe474d7a12` with the
verified local FDR26 archive, readiness and DeepDRiD witnesses, and APTOS
checkpoint and source preflight inputs bound in the reports. The replay and
evaluation use four CPU threads. Run the audit after scoring.

```text
python scripts/evaluate_fdr26_aptos.py --source-commit 799f70874b0fab00284f1dcb31d9e4fe474d7a12
python scripts/audit_fdr26_aptos.py
```

FDR26 v3 is attributed to Sanae Lamti, Noureddine Kerzazi, Rachid Sekhsoukh
and Asmae Guechati. [Author deposit](https://doi.org/10.5281/zenodo.23101200) ·
[Data article](https://doi.org/10.1016/j.dib.2026.113379) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
