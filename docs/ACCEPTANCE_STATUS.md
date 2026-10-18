# Acceptance evidence — 18 October 2026

## VERIFIED COMPLETE — scoped engineering pilot

- Existing assets audited and preserved; dedicated local project and Git history established.
- Official DeepDRiD metadata and terms inspected; 260 real images acquired with provenance.
- Patient-separated train/selection/calibration/holdout manifest validated against actual files.
- Corruption, missing-label, source ambiguity, subject overlap and duplicate checks exercised.
- Two MobileNetV3-small models genuinely trained on 160 images for three CPU epochs.
- Checkpoints selected on validation loss, loaded on CPU with schema/checksum validation.
- Binary quality pathway and temperature scaling fitted on 20 separate calibration images.
- Three actions, independent integer cohort capacity limits and explicit infeasibility verified.
- Horizontal-flip second pass with separate calibration and exact activation capacity verified.
- Actual held-out pilot grading, calibration, seven policy comparisons, natural quality/severity
  tables, operating points, ablation availability/infeasibility, and patient bootstrap recorded.
- Five CPU strategies timed on validation images, with hardware and timing limitations.
- FastAPI upload, readiness, typed responses, batch limits and privacy-conscious errors tested.
- Five Streamlit pages rendered; actual HTTP real-image result rendered with export widgets.
- Full test run with real-image opt-in: 78 passed, one upstream deprecation warning.
- Ruff checks passed. Environment, source and result provenance recorded locally.
- Docker CPU image built and both services reported healthy. Actual real-image
  HTTP prediction and batch/TTA passed against the containers; readiness, model
  info, docs and dashboard health returned 200. Container source hashes match
  all 27 local runtime Python files. See reports/deployment.json and
  reports/container_source_verification.json. Local ports are 18000/18501.

These checks establish execution, not screening usefulness. The pilot DR model predicted
grade 3 for every holdout image: accuracy 16.67%, macro-F1 0.0571, kappa 0. Quality
predicted all images poor (50% accuracy). Most capacity settings were infeasible.
At review 0.6 / recapture 0.4, full deferral gives zero automated coverage; missed-auto
counts cannot be interpreted as screening success. One joint allocation sent more severe
cases to recapture than simple policies. No settings were retuned using this holdout.

## IMPLEMENTED BUT NOT VERIFIED

- Optional pretrained initialization, optional GPU/mixed precision and deployment on other
  operating systems/hardware have not been exercised as scientific experiments.
- General quality multi-head support exists, but the initial real model trains the overall
  binary quality label only; mapped natural-defect heads have not been genuinely trained.

## PARTIALLY COMPLETE

- Research evaluation: exploratory pilot artifacts exist. They cannot substantiate the
  proposed contribution. Many ablations correctly report infeasibility with weak scores.
- Literature: 43 primary-linked records, 43 BibTeX entries, 28 original IDs retained, with
  record-specific verification limits. This is not a replication of all cited papers.
- Paper: structured manuscript draft and methodology exist; confirmatory results remain open.
- Review 1: actual progress brief and architecture source exist; original Office materials
  are preserved, not overwritten with stronger claims.

## BLOCKED OR NOT STARTED — final scientific acceptance

- Useful model validation, full-size training and backbone comparison: NOT STARTED beyond pilot.
- Independent external-domain experiment: NOT STARTED. IDRiD source access/terms could not
  be retrieved by the audit; EyePACS needs its own terms; DDR author archive requires a
  separately planned larger acquisition. None is replaced with synthetic clinical evidence.
- Full natural defect training, controlled degradation experiments and complete ablations:
  PARTIALLY COMPLETE in software; final real research experiments NOT STARTED.
- Matched realized-workload superiority, paired patient-level statistical comparisons and
  a defensible publication contribution: NOT ESTABLISHED.
- Clinical validation, actual specialist diagnoses and paired recapture benefit: NOT STARTED.

Next executable research task: predeclare a larger model-development experiment on the
remaining official training patients, train with an appropriate image resolution and
optional licensed pretrained initialization, select using development data, and reserve
a new untouched confirmatory cohort. Do not turn the consumed pilot holdout into a tuning set
while still calling its performance final.
