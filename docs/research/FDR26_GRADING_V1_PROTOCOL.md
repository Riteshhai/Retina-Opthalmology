# Frozen DeepDRiD grading on FDR26 v3

Evaluate the selected DeepDRiD DR checkpoint against the released ALFEDIAM
grade codes on FDR26 v3. The five source stages are aligned by nominal code
for this diagnostic. ALFEDIAM and ICDR differ in lesion criteria. The
binary reference is ALFEDIAM grade 2 or higher. The fixed prediction rule
uses the sum of model probabilities for codes 2–4 and a threshold of .50
with inclusive ties. This endpoint measures agreement with the source
grades. Clinical referral, image quality and macular edema are outside scope.
[Author deposit](https://doi.org/10.5281/zenodo.23101200) ·
[Data article](https://doi.org/10.1016/j.dib.2026.113379) ·
[Classification comparison](https://www.has-sante.fr/jcms/c_1028304/en/argumentaire-depistage-de-la-retinopathie-diabetique-par-lecture-differee-de-photographies-du-fond-d-oeil).

Bind the FDR26 readiness report, independent receipt, private witness and
photo archive before scoring. Use the 2,849 eligible photos as the primary
cohort. Report the 2,742 fingerprint representatives, the 2,761 photos from
the extended filename-prefix exclusion and their 2,673 representatives as
sensitivity views. Each photo is scored once. The published cohort is
grade-enriched and the filename groups are unconfirmed. Report image-level
point estimates. A fingerprint screen against the frozen APTOS and DeepDRiD
references cannot establish patient independence.
[Readiness audit](FDR26_READINESS_V1_RESULTS.md).

Use the selected 224-pixel DR model, native crop and letterbox, RGB
normalization, two CPU inference threads and single-image first-pass
forwards. Record raw logits and raw and source-temperature-scaled softmax
probabilities. The first-pass temperature was fitted on DeepDRiD and is
fixed. This study fits no model or temperature and selects no threshold
or replacement checkpoint. Keep per-image outputs local.

Report accuracy, macro-F1, quadratic-weighted kappa, mean absolute grade
error, the count of undercalls by two or more grade codes, and the five
grade denominators, confusion matrix and recalls. Report NLL with probability
floor 1e-12, summed multiclass Brier and top-label ECE with ten equal-width
bins. For the binary grade endpoint, report AUROC, stepwise average
precision, confusion counts, sensitivity, specificity and the accuracy
of a fixed always-positive reference. Compute binary scores by summing
the stored probabilities in float64. Report raw and calibrated results.

The independent audit will reconstruct all populations and metrics and
verify all stored probabilities against logits. It will replay the first
two eligible photos per grade in readiness witness order on the same CPU
path, checking logits to absolute tolerance 1e-6.
