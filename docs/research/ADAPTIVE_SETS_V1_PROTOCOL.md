# Fixed APS/RAPS development comparison — 6 November 2026

This protocol is declared before new APS/RAPS outcomes. Prior APTOS grading,
LAC/ordinal/calibration outcomes and consumed IDRiD diagnostics are already known.
This is a saved-prediction source-development diagnostic under MASTER_SPEC_v1.0,
not fresh confirmation, clinical validation or an exact paper reproduction.

## Sources and implementation boundary

Use the ranked-mass construction in [Angelopoulos et al., arXiv v5](https://arxiv.org/html/2009.14193v5).
Sort probabilities descending with lower-grade ties. For rank r and one explicit
uniform u per observation, APS score is preceding probability mass plus (1-u)p_r.
RAPS adds lambda max(r-k_reg,0). The same u is used across all labels of a row.
Fit the exact ceil((n+1)(1-alpha)) order statistic, or infinity if rank exceeds n;
include equality. RAPS scores/thresholds may exceed one and must not be clipped.
Return raw sets and common argmax fallback separately; enlargement can add a label.

The [retinopathy author source](https://github.com/umarbhasan/conformal-diabetic-retinopathy/blob/3a7d18d799262098d8166519159d538960bbf118/src.py)
was rechecked as primary source evidence. Its initial CFG uses k_reg=2, lambda=.10;
the later code tunes RAPS and refits on the full calibration role. This study fixes
the initial values before outcomes and performs no parameter tuning. It shares
uniforms across methods and retains the exact conformal order statistic, differing
from the authors' sequential random draws and NumPy quantile indexing. Author code
is neither imported, executed nor copied into QSART. Full 384px/five-model-seed/
tuned-procedure reproduction remains incomplete. Production sidecar schemas stay fixed.

## Frozen population, model and reuse

Source report reports/aptos_optimization_v2/evaluation.json SHA256
66c49f12bf630493c9dc38180a73daeeae6a222523067f4ec262971b45d2310e. Keep its selected
primary training seed2026, epoch10 EfficientNet-B0 and temperature2.940995751533916.
Verify the checkpoint and both saved prediction CSV hashes, runtime manifest/roles,
exact row/group/label alignment and representative flags. Read no reserved test
manifest labels, photographs or new neural predictions. Source labels/partitions,
quarantine and unknown patient-grouping limitations remain unchanged.

The 262 historical temperature images remain outside conformal fitting. Use exactly
254 conformal representatives and all 521 reused selection observations; retain
the 507 selection representatives as a second descriptive population. Verify no
image ID or fingerprint component crosses the calibration/selection roles, and
one representative per component. Neither component grouping nor grade-stratified
splitting proves iid patients or exchangeability. Calibration and selection were
previously examined; no fresh calibration or external-validity claim is made.

## Complete fixed comparisons

Alpha .10/.05. LAC and deterministic ordinal once per alpha; randomized ordinal,
APS and fixed RAPS for uniform seeds2026–2030. Reuse the original SeedSequence(seed)
spawn(2) streams in original calibration/selection row order, with common uniforms
across randomized methods. These five seeds are randomization replicates of one
fixed trained model, not model training uncertainty. This gives34 score comparisons,
each raw/fallback and all-images/representatives:136 aggregate metric views and
35,428 per-image set rows retained privately. Reproduce all14 historical LAC/ordinal
controls, both empty policies and both populations, allowing threshold roundoff at
1e-12 but requiring identical measured metrics.

Report every threshold, rank, inclusion/support by grade, mean/size distribution,
empties, full sets and nonempty-contiguous fraction. No best method, seed, alpha,
parameter, sidecar or production replacement is selected. APS/RAPS rank sets can
be noncontiguous in disease-grade space; ordinal adjacency is a structural property.

## Primary and paired development analysis

Primary: alpha .10, uniform seed2026, raw inversion, RAPS minus APS on all521
selection rows. Jointly inspect mean set-size change and marginal inclusion change;
smaller sets alone do not establish improvement. Keep grade3/4 inclusion visible.
Secondary: randomized ordinal minus APS. For both contrasts, all five uniform seeds,
both alphas and raw/fallback, retain paired changes in mean size, marginal inclusion
and grade3/4 inclusion:40 contrast records/160 descriptive95% intervals.

Use2,000 common whole-fingerprint-component draws, bootstrap seed20261109. Each
draw retains all members and repeated components. Report the valid draw count for
each grade endpoint; absent-grade draws are excluded only from that endpoint.
Condition on fixed classifier, temperature, thresholds and uniforms; no refitting,
reselection, patient-independence inference or multiplicity correction. The
intervals cannot validate clinical triage, shifted-domain or grade coverage.

## Execution, preservation and verification

Default runner verifies saved inputs only. Commit tested source/config/protocol
clean before --execute creates new immutable public aggregates and private sets.
Preserve prior master/reports/weights/source manifests, all EyeQ evidence and input
logits. Do not overwrite measured outputs or original datasets. No image decoding,
classifier/temperature fitting, RAPS tuning, IDRiD/EyeQ/APTOS-test inference or
production update occurs in the new study.

Independently reconstruct ranked/ordinal candidate scores, exact thresholds, every
raw/fallback set, all metric views and160 intervals from saved logits using a
separate verifier, not the new adaptive module. Verify 14 historical controls,
source provenance, preservation, public privacy and unchanged genuine deployment.
Test hand-calculated fixtures, ties, lambda-zero APS equivalence, thresholds above
one, exact rank/empty/infinity boundaries, malformed artifacts and role separation.
Exercise actual API/UI controls and native/available Docker services with trained
models. Plot only saved measured results and visually inspect exported figures.
Formal manuscript drafting remains deferred.
