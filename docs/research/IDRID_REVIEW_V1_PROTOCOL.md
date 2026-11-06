# IDRiD source confirmation and overlap review v1 — 4 November 2026

The user has now confirmed the official IDRiD/IEEE DataPort download. Preserve the
original data-audit report, whose origin status was unconfirmed when executed.
Append a source-confirmation receipt bound to its report, manifest and image-
content hashes. Source testimony does not authenticate an official archive checksum
or resolve the observed duplicate/label-conflict findings.

This follow-up inspects ONLY the previously declared 8 internal and 3 cross-source
candidate pairs. No new distance thresholds or outcomes are selected. Commit the
tested comparison/population code and this protocol before real pair review.
Reverify every touched image against its frozen SHA256. Resolve APTOS images from
the relocated configured source, and IDRiD images from its fixed private manifest.

For each pair record original-byte equality and EXIF-oriented RGB pixel equality.
An exact decoded-pixel match proves identical displayed image content, without
establishing patient identity or correcting its grade. Otherwise retain the pair
as an unresolved perceptual candidate. Also record descriptive 224px canonical
RGB equality, mean absolute error, RMSE and correlation using existing fundus
geometry. These normalized comparisons never upgrade a pair to an original-pixel
match, establish same-eye identity, or determine an inclusion mask.

Retain source DR/DME disagreements without voting, relabeling or clinical
adjudication. Preserve all516 source observations and the original manifest.
Freeze a new private full-row population table with three explicit views:

1. Full source observations, including duplicates and ambiguous annotations.
2. Conservative cross-source filtering: flag every IDRiD component containing
   ANY original APTOS candidate, including unresolved candidates. No new label,
   pixel-distance or model-outcome decision changes this rule.
3. One representative per retained fingerprint component, chosen by the minimum
   SHA256 of `idrid-review-v1:` plus namespaced source identity. This rule uses no
   labels, row order, model predictions or accuracy. Retain ambiguous-component
   flags even when its representative has one source grade.

These are sensitivity populations, not an asserted independent-patient cohort or
an externally selected winning population. All rows remain external_holdout with
eligible_for_training=False. Full and filtered results must be reported under a
later locked evaluation protocol; this step computes NO classifier prediction,
fit, temperature, conformal threshold, model selection or performance metric.

Public aggregates record comparison categories, all frozen population counts,
unresolved label conflicts, source/report/input/output hashes and limitations.
Image paths, identities, pair metrics and population membership remain ignored
under artifacts/idrid_review_v1. Refuse overwriting completed or partial outputs.
Independently reproduce decoded-pixel evidence and all membership choices; verify
source testimony appears in the real API/UI while the historical audit and all
production/master/previous study bytes remain unchanged. Formal writing stays
deferred and useful clinical/triage validity remains unestablished.
