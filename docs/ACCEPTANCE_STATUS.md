# Acceptance evidence — 20 October 2026

> Preserved v1 evidence. The current MASTER_SPEC_v1.0 alignment, policy-version2
> behavior, conformal limitations and deployment checks are recorded in
> [SPEC_V1_STATUS.md](SPEC_V1_STATUS.md). Historical AUTO_SCREEN results are not
> current AUTO_NONREFERABLE policy results.

## Verified engineering and internal research milestone

- Original Office assets, literature library, starter and pilot evidence preserved.
- Pinned DeepDRiD photos, metadata, source terms and per-image checksums recorded.
- Development: 1,184 images / 296 patients, with fitting 848, selection 168 and
  calibration 168. Ambiguous subjects excluded; patient and fingerprint groups
  remain together. Fresh confirmation: 340 images / 85 patients, excluding all
  inspected pilot holdout patients. Cross-cohort image audit passed.
- Declared ImageNet transfer protocol executed on CPU at 224px. Frozen MobileNet
  and ResNet18 DR candidates compared on selection only; MobileNet selected.
- Binary quality and three mapped natural-defect heads genuinely trained. Fixed
  classical statistical quality comparator fitted on training data only.
- Dedicated calibration and pre-confirmation artifact/source lock completed.
- Internal confirmation: 60.0% DR accuracy, 0.532 macro-F1, 0.726 QWK, 0.925
  referable AUROC; accuracy patient-cluster interval 52.6–67.1%. Overall-quality
  accuracy 74.1%, versus 73.2% classical. Defect heads remain weak; artifact-head
  accuracy falls below the constant majority comparator.
- All seven policies across nine budget pairs reported; 27/63 cases infeasible.
  Matched-realized-workload comparisons use exploratory patient-cluster intervals,
  with no confirmation tuning. Review and recapture remain unresolved outcomes.
- Five CPU strategies executed on 48 selection images, three repeats, two CPU
  threads. Sequential live-workstation timings do not establish a speed advantage.
- API, five-page dashboard, calibrated inference, integer capacity enforcement,
  upload rejection and temporary upload handling verified. Dashboard recorded
  metrics require a model card matching both loaded weights and calibration.
- Full suite with new real-model opt-in: **89 passed**, 39.07 seconds; one upstream
  AnyIO deprecation warning. Ruff passed.
- Docker rebuilt and actual new-model HTTP readiness, single/batch/second-pass,
  upload rejection and dashboard health passed. Local ports: **18000 / 18501**.
  See `reports/deployment_v1.json`; service readiness does not certify accuracy.

Current evidence: [results and limits](research/CONFIRMATION_V1_RESULTS.md),
[frozen report](../reports/confirmation_v1/evaluation.json),
[model card](../reports/development_v1/model_card.json), and PROGRESS.md.
Original pilot results remain in reports/pilot and the historical execution log.
The new and old cohorts/protocols differ; no paired improvement estimate is made.

## Partially complete — overall research project

The literature audit retains 43 primary-linked records and all 28 original IDs,
with record-specific limits. The manuscript, methodology, current Review-1 brief
and architecture source exist. Source, scripts and aggregate evidence are kept
in the GitHub draft PR; datasets, identifiers, per-image artifacts and weights
remain local. Internal results support further study, not publication readiness.

## Remaining evidence gates

- Independent external-domain validation and overlap/label harmonization.
- Independently tuned comparators, repeated seeds, expanded model development,
  controlled degradation experiments and full robustness ablations.
- Reliable natural-defect prediction; small/absent defect strata limit inference.
- Patient-level referral logic, DME/other disease handling and reader uncertainty.
- Prospective workflow evaluation, actual specialist diagnoses, paired recapture
  benefit and clinical authorization: not established.
- Optional GPU/mixed precision and other hardware/OS deployments: not exercised.

The overall acceptance remains **PARTIALLY COMPLETE**. Future model/policy tuning
must acknowledge the now-inspected confirmation population and reserve independent
evaluation data. No clinical benefit or universal policy superiority is claimed.
