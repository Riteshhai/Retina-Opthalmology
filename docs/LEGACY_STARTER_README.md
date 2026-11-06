# QSART — Research Starter v0.1

**Status:** Working OFFLINE policy-evaluation proof of concept; no trained retinal image classifier or validated medical device. Built-in predictions are entirely synthetic. This package is not clinically usable.

## Why this exists
The 10-credit thesis seeks to test if jointly accounting for estimated image gradability, DR severity risk and uncertainty reduces **dangerous auto-clearances** while also auditing the risk of delaying severe cases by assigning recapture rather than human review while controlling specialist-review and recapture demand. NaIA-RD (2025) and 2026 retinal triage studies already have similar high-level components; three-way routing itself is not novel.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m qsart.cli --demo --output results/mock
streamlit run app.py
```
Docker (if installed): `docker compose up --build` then open http://localhost:8501.

## CSV schema
`image_id,patient_id,grade,quality,p_rdr,p_severe,uncertainty`
- grade 0..4 (retrospective label; never input to the policy).
- quality 0..1, **higher means more gradable**.
- p_rdr = hypothetical calibrated probability for grade >=2.
- p_severe = hypothetical probability for grade >=3.
- uncertainty 0..1 (higher = more uncertain; not automatically calibrated).

**Important:** The package never creates medical predictions from photographs. Real scores must later come from properly trained, held-out, independently tested models. Do not insert synthetic results in any paper as clinical findings.

## Evaluation warning
`cli.py` independently tunes QSART and each baseline using calibration-only constraints, with infeasible comparators explicitly reported. Matching **maximum** review/recapture budgets does not necessarily match actual workloads. The research paper MUST additionally match operating points, compare on a locked test set, run patient-grouped confidence intervals, and include any infeasible operating points.

## Leakage safety
`cli.py` rejects patient ID overlaps between calibration and test. You are responsible for checking duplicate eyes, near-duplicate images, other dataset overlaps, and provenance. For shared EyePACS/EyeQ imagery, never treat them as independent cohorts.

## Next code milestones
1. Native DeepDRiD CSV/image loader with true per-patient grouping and source-domain metadata.
2. Train actual compact DR and quality predictors with class-balanced transfer learning.
3. Export calibration and test predictions to the CSV schema.
4. Fair independent tuning of each baseline, bootstrap by patient, quality strata, paired tests, predefined constraints.
5. ONNX export, CPU and Docker benchmark with real models, image upload, audit logging, deployment warning.

Refer to docs/RESEARCH_PROTOCOL.md, docs/LITERATURE_MATRIX.md, docs/IMPLEMENTATION_ROADMAP.md.

## Verified real dataset metadata audit (no model training)
The official DeepDRiD CSV uses `patient_id,image_id,image_path,Overall quality,left_eye_DR_Level,right_eye_DR_Level,patient_DR_Level,Clarity,Field definition,Artifact`. **Use the correct EYE grade based on `_l1/_l2/_r1/_r2`, not `patient_DR_Level` as the per-image target.** The official labelled train/validation splits contain 1,200 and 400 image records respectively. Keep patients separate. Source CSVs:
- https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/master/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv
- https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/master/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv

After obtaining the data from its official project, run:
```bash
python -m qsart.deepdrid \
 --train data/raw/DeepDRiD/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv \
 --validation data/raw/DeepDRiD/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv \
 --out results/deepdrid_audit.json
```
The audit checks eye-level labels, patient overlap, patient counts, grade/quality cross-tabs, and missing quality components. It **does not** download images. Match dataset license/terms separately.
