# Adoption of MASTER_SPEC_v1.0 — 20 October 2026

The user supplied the updated specification and requested alignment with completed
work. MASTER_SPEC.md is the authoritative current copy; the exact versioned copy
is docs/specifications/MASTER_SPEC_v1.0.md. Both preserve the uploaded text bytes,
SHA256 1b6af695af482d43d763e7e5cb74d4fe1002f3778fba8abc55872500a5a8c275.
The previous build prompt, original Office assets and all measured results remain.

## Reuse and requirement gaps

| Requirement | Verified existing evidence | Alignment work / current status |
| --- | --- | --- |
| R1/R2 real working AI | 1,184-image development, frozen ImageNet MobileNet and ResNet training; 340-image internal confirmation | VERIFIED COMPLETE for current DeepDRiD model pipeline; EfficientNet DeepDRiD candidate completed; APTOS study BLOCKED |
| R3 CPU Docker/API/frontend | Built and real JPEG single/batch/TTA verified, including restart; 31 runtime source files matched | VERIFIED COMPLETE for v1; every new integration must be rebuilt and rechecked |
| R4/R5 cost/CPU | Local Python 3.11, CPU PyTorch; no paid services | VERIFIED COMPLETE on this workstation; other laptop configurations unmeasured |
| R6/R8 integrity/reproducibility | Frozen manifests, real outputs, source/weight hashes, seeds, aggregate reports | Preserve v1; separate new spec experiments from consumed confirmation |
| R7/N1 primary triage | Independent exact capacities and seven policies exist | IN PROGRESS: AUTO_NONREFERABLE replaces provisional AUTO_SCREEN semantics; entropy-only baseline, independent policy tuning and simulator required |
| R7/N2 quality × severity | Genuine multihead quality, classical comparator, natural strata/intervals | Existing analysis verified; controlled degradation and expanded ablations NOT STARTED |
| R7/N3 conditional CPU | Actual five-strategy flip-TTA benchmark, with no speed advantage claim | Existing implementation verified; complete end-to-end timing and new decision semantics still required |
| Uncertainty / base paper | Confidence, entropy, temperature, Brier/ECE/reliability exist | IN PROGRESS: primary-paper critical study; conventional/ordinal conformal with independent calibration |
| Dataset roadmap | DeepDRiD acquired and audited | APTOS official access unresolved; IDRiD held out, EyeQ/DDR/Messidor optional until access/label audit |
| Product modules / endpoints | Five pages; real upload, batch, exports, saved analytics and model information | IN PROGRESS: simulator, comparison, experiment explorer and typed research/config endpoints |
| R9 demonstration | Real image → trained predictions → v1 triage → export verified | Must verify new non-referable/conformal workflow before claiming alignment complete |
| R10 paper readiness | 43 primary-linked references, prior methodology/draft/evidence | IN PROGRESS overall; formal writing deferred; publication contribution unestablished |
| R11/R12 responsibility | Research warnings, missing artifacts rejected, no generated prediction fallback | Preserve and test under every new route and artifact type |

## Architectural decision CR-001: action semantics

The new user specification explicitly changes the automatic action to
AUTO_NONREFERABLE. This is a behavior change: the historical AUTO_SCREEN action
could carry any provisional grade, including referable disease. Existing reports
cannot be relabeled as a non-referable triage evaluation.

Implement the current policy in a new module, with mandatory review for predicted
referable/high-risk cases and restricted automatic eligibility. Poor quality,
uncertainty and conformal ambiguity prevent automatic clearance under configured
research gates. If the configured capacities cannot accommodate required
deferrals, return an explicit overload state with safe aggregate counts.
Thresholds are operational assumptions, not validated clinical safety cutoffs.

The public current API will declare policy version 2.0. Old research allocation
source remains available for replaying v1. Client action names change explicitly;
no positive output is silently converted to non-referable. New metrics count any
true referable AUTO_NONREFERABLE as a miss. Deferred cases remain unresolved.

## Architectural decision CR-002: conformal sidecar

Preserve the currently evaluated DR and quality predictions. Fit an independent
DR temperature for conformal scores on 21/42 calibration patients (84 images),
then fit conformal scores on the other 21 patients (84 images). The split is
fixed by subject hash and seed before new fitting. Existing checkpoint selection
used neither subgroup. The conformal sidecar explicitly identifies its independent
temperature transformation; it does not reuse all-168 fitted temperatures for
conformal calibration while claiming an independent fit.

Account for all correlated views with optional patient-maximum scores. Conventional
LAC and a documented paper-inspired ordinal construction are separate methods.
First-pass and all-flip paths have separate artifacts. Conditional TTA routing,
severity stratification and external shift do not inherit an unconditional formal
coverage claim. Missing/mismatched configured sidecars fail honestly.

## Compatibility and work order

The sprint starts from verified work, not day-one scaffolding. Keep the working
qsart package layout instead of a disruptive src/ migration. Compose defaults stay
8000/8501; local overrides 18000/18501 avoid the unrelated application on 8000.
Retain the selected genuine MobileNet until a real candidate wins a declared
comparison; an EfficientNet default in a specification is not a trained checkpoint.

Current source before adoption is cedf182846845a782bb6e30336d9c8f99b2e31fb.
An ignored local Git archive preserves exact v1 source before extension; archive
SHA256 7fc86379169d42069553161f618d968a7e5ea83cc78cc33908a46017bfbb180a.
The historical lock remains a record of the old experiment, not a mutable current
source identifier. Replaying it requires the preserved revision.

First address the changed image-to-decision contract, add conformal and simulator
integration, exercise real development images and rebuild Docker. Then pursue
EfficientNet/APTOS, independently tuned policies, repeated seeds, controlled
degradation and external evaluation under newly declared protocols. No fresh
unbiased confirmation claim will use the consumed v1 population.


## Executed alignment outcome — 21 October 2026

The new action contract, conformal sidecar, all eight dashboard modules and typed
research/simulation endpoints have been exercised. The full integrated suite
passed 198 tests, including the selected genuine models and CP, and Ruff passed.
The rebuilt Docker service passed real JPEG single/batch/TTA, matched artifacts,
simulator evaluation/overload and research routes.

The EfficientNet-B0 candidate is genuinely trained on DeepDRiD, not APTOS:28 epochs,
756 updates, selectedepoch 13, selection64.29% accuracy and0.5249macro-F1. MobileNet
remains selected by the declared macro-F1 rule. The user confirmed APTOS is not
locally downloaded. Its adapter is software-tested, not real-data validated.

Independent CP fitting used84 images/21 patients for temperature and a disjoint
84/21 for conformal, with no photo/fingerprint overlap. On168 selection images,
LAC image coverage94.64% and mean setsize 3.345 coexist with patient-all-views
coverage85.71%. The deterministic ordinal predictor returns all 5 grades for every
image. All72 feasible cases of 360 policy/profile/budget combinations review every
image;288 cases overload. This is an uninformative current ordinal triage result,
not useful automatic screening or a successful novelty claim. No thresholds were
changed to hide it. See docs/SPEC_V1_STATUS.md for the complete current status.
