# QSART Development Status — updated 9 November 2026

- **Current specification:** MASTER_SPEC_v1.0, exact root/versioned copies retained.
- **Verified pre-adoption commit:** cedf182846845a782bb6e30336d9c8f99b2e31fb;
  new implementation source and aggregate evidence are delivered on the same
  qsart/research-prototype branch/draft PR. Exact v1 source is preserved locally.
- **Current milestone:** specification alignment and integrated real-image
  execution VERIFIED COMPLETE; overall scientific project IN PROGRESS.
- **Current training extension:** the matched 224/384-pixel APTOS comparison is
  VERIFIED COMPLETE from `cc4f4450cb9b2663def6cb1e637949cdfdd50782`. The 224-pixel
  arm completed 15 epochs and selected epoch 12 at 83.11% accuracy and 0.715
  macro-F1. All four paired intervals for the selected and epoch 5 contrasts
  include zero. The audit reconstructed all 14 conformal settings and verified
  350 preserved artifacts. All 631 local tests and live Docker checks passed.
  Multiple training seeds and independent validation remain open.
  [Matched results](research/APTOS_MATCHED_RESOLUTION_V1_RESULTS.md).
- **Previous training extension:** the locked 384-pixel APTOS development run is
  VERIFIED COMPLETE from `a446d647535d629f03f46ea2b43a73e29523fff7`. It selected
  epoch 9 after 13 epochs and 3,445 updates. Selection accuracy was 83.49%,
  macro-F1 0.697 and grade 3 recall 8/26. Both paired intervals against the prior
  224-pixel run include zero. The audit verified 340 preserved artifacts and
  reconstructed all 14,588 saved conformal sets. All 627 local tests passed.
  The rebuilt Docker app passed real-image checks and serves the research
  report. [384-pixel results and verification](research/APTOS_384_V1_RESULTS.md).
- **Previous data extension:** pinned user-selected Hugging Face EyePACS cache and
  independent compatibility audit VERIFIED COMPLETE from recovery source
  `c1e52fd21f43e38de78007e1376055276be5865d`: 14 files/6,535,643,877 bytes/35,108 unique
  rows, all file/card/receipt hashes verified. Exact author annotation joins yield
  12,529 train matches, fourteen missing Reject-labelled train rows and zero of
  16,249 test images. Shared DR labels agree. All 307 prior artifacts remain fixed.
  Transformed quality truth, original terms/pixels and patient independence remain
  unverified; both EyeQ partitions remain external-only and evaluation-ineligible.
  [Acquisition results](research/EYEPACS_HF_V1_RESULTS.md).
- **Current feasibility extension:** bounded fitting-only 224/384 CPU probe VERIFIED
  COMPLETE from the same source. 32 fitting images/four actual AdamW updates per
  fresh worker, finite loss/gradients, no saved candidate/predictions. 224/384 update
  times 2.71586s/8.24752s and sampled process-tree RSS 1.310 GB/2.742 GB. These are short
  concurrent-workload observations, not full-training resource or accuracy claims.
  [Measured feasibility](research/APTOS_384_FEASIBILITY_RESULTS.md).
- **Previous score extension:** fixed APS/RAPS development comparison VERIFIED COMPLETE
  from clean `e7fd2f1b85d79e2f9f910433eea0dd7beea06b7c`. All 34 thresholds,
  136 aggregate views, 35,428 raw/fallback set rows and 160 paired intervals are
  independently reconstructed. Fourteen historical LAC/ordinal controls reproduce;
  295 prior artifacts remain fixed. Primary RAPS mean size 1.53935 versus APS 1.63916
  coexists with grade-3 inclusion 53.85% versus 69.23%. One reused APTOS source
  model/temperature, fixed initial parameters and uniform replicates establish no
  fresh, grade-specific or clinical guarantee. Production remains unchanged.
  [Fixed APS/RAPS results](research/ADAPTIVE_SETS_V1_RESULTS.md).
- **Previous metadata extension:** EyeQ annotation/access audit VERIFIED COMPLETE from clean
  `9f7f22f79362090039369f8a34625d326189c6e2`. All 28,792 rows, 20,662 filename
  subject keys, class/DR matrices, eye groups and four source-table comparisons
  independently reconstructed; all 271 prior artifacts remain fixed. Both author
  splits are external-only. Native and rebuilt Docker services expose 19 records
  and pass actual trained-model checks, with 65 container source/config files
  matched. Zero EyeQ images decoded or model outcomes produced; photograph/terms/
  overlap/harmonization readiness remains incomplete.
  [Metadata audit results](research/EYEQ_DATA_V1_RESULTS.md).
- **Previous calibration extension:** fixed-model quality calibration/stress VERIFIED COMPLETE
  from clean `42ae8621649b11d666f7b5e3e8516b0ac3a5a812`:24 scalar temperatures,
  six candidates, seven conditions,8,064 model-image observations. Primary clean
  mean NLL 1.02046 to .87412, but field ECE/Brier and frozen-model mean NLL worsen.
  Every class prediction stays fixed. All 486 intervals/48 zero controls,96 clean
  head metric views and 234 preserved artifacts independently reconstructed;
  all logits replay exactly. Stress score changes have no new quality truth and
  cannot establish detection accuracy or triage utility. [Results](research/QUALITY_CALIBRATION_STRESS_V1_RESULTS.md).
- **Previous quality extension:** matched quality-backbone adaptation VERIFIED COMPLETE
  from clean `2d4243c977a0669776a5dc356d6d6e3166732a12`: three seeds, two arms,
  fixed15 epochs/795 updates per run. Conditional mean four-head F1 .50894 frozen
  versus .54112 adapted; all three primary paired intervals include zero. Raw
  calibration and poor-quality AUROC worsen, and rare-class failures remain.
  Independent reconstruction covers90 trained/6 warm metric sets,27 intervals,
  1,016 source bytes and171 preserved artifacts. Version1.1 corrects paired-row
  seed metadata only; original report/model/epoch/prediction/metric/interval bytes
  remain preserved. [Quality results](research/QUALITY_ADAPTATION_V1_RESULTS.md).
- **Previous source extension:** primary-seed APTOS schedule extension VERIFIED COMPLETE
  from `a75229f92468a253e8f6c3e28d834571e25bd9e5`. One fixed primary seed 2026,
  stopped after 14/15 epochs, selected 10. Selection accuracy 80.81%,macro-F1 .65157;
  grade 3 recall 26.92%/grade 4 67.50%. All four paired intervals include zero.
  All 14 uncertainty comparisons/14,588 sets and 134 preserved artifacts audited.
  LAC alpha .10 grade 3 inclusion 46.15% remains poor. The owned-mask warning fix
  followed measured execution/audit without changing any scientific bytes.
  [Source development results](research/APTOS_OPTIMIZATION_V2_RESULTS.md).
- **Completed external extension:** locked IDRiD external-domain evaluation VERIFIED COMPLETE
  at `e447a21b55b82ca47d1e3b91caf76bca61f23699`. Four frozen source classifiers,
  all 56 uncertainty comparisons and all 516/full,513/filtered,505/representative
  views are retained. Primary 2026/main 505 accuracy 42.97%, macro-F1 .41359; source-T
  LAC alpha .10 inclusion 70.69%. Independent reconstruction verified 57,792 sets,
  360 annotation envelopes and 18 intervals. IDRiD model outcomes are now consumed;
  no further use may be disguised as fresh confirmation or used for final tuning.
  [External results](research/IDRID_EXTERNAL_V1_RESULTS.md).
- **Completed source-data review:** IDRiD source confirmation, pixel evidence and
  population freeze VERIFIED COMPLETE at `d1c5968189f8bd36508b5cbb4c35220af5f0bf6b`.
  Full 516/filtered 513/representative 505 views retain original grades and all four
  ambiguous components. Source origin is user-confirmed; archive checksum is not
  independently authenticated. Three cross-source candidates remain unresolved;
  three byte-identical internal pairs have conflicting DR labels. No external
  classifier outcome existed at that review stage. [Review results](research/IDRID_REVIEW_V1_RESULTS.md).
- **Initial IDRiD data audit:** integrity/overlap execution VERIFIED COMPLETE from
  `d597daa553b58c57f8e96f1b33a2f24bf8a9e4f8`. All 516 grading JPEGs decoded;
  independent reconstruction reproduced 132,870 internal and 2,675,976 reference
  distances. Three perceptual APTOS matches and four internally label-conflicting
  components remain unresolved. The user has since confirmed the official source;
  a bound receipt preserves the original audit-time status. All images remain
  external-only; classifier evaluation had not started at that audit stage, and an unqualified
  independent-cohort claim is blocked by these findings. See
  [audit results](research/IDRID_DATA_V1_RESULTS.md).
- **Completed APTOS extension:** bounded three-seed parameter-fine-tuning and
  independent audit VERIFIED COMPLETE. All 3 runs/42 uncertainty comparisons
  executed from clean `8357e9d918ef9ec00d1f66e4efadc3325790097c`.
  Conditional macro-F1 0.62411 ± 0.00823 sample SD;
  predeclared primary seed 2026 0.63349. Its grade-3 recall improves to
  65.38%, while grade-4 recall falls to 47.50%.
  See [measured results](research/APTOS_FINETUNE_V1_RESULTS.md) and the preserved
  [frozen-feature study](research/APTOS_V1_RESULTS.md). Production stays unchanged;
  convergence, exact-paper reproduction and fresh-test utility remain unproven.
- **Earlier degradation execution:** `493ad7c6e37d16dc338e00280d8ac714ee23a0c8`;
  fixed seven-condition selection-data degradation VERIFIED COMPLETE. Full
  [results and limitations](research/ROBUSTNESS_V1_RESULTS.md) retain the observed
  blur/compression failures and non-monotonic quality response.
- **Further verified executions:** ordinal diagnostics at `16bd809` and six real
  repeated-seed training runs at clean `99a6cbb`. All fixed outcomes are preserved;
  see [ordinal study](research/ORDINAL_STUDY_V1_RESULTS.md) and
  [training variability](research/REPEATED_SEEDS_V1_RESULTS.md).

## Implemented and actually tested

Current AUTO_NONREFERABLE / RECAPTURE / SPECIALIST_REVIEW policy enforces
referability, severity, quality, uncertainty and supplied conformal guards under
independent integer capacities. Inadequate capacity yields structured overload;
there is no forced clearance. Old AUTO_SCREEN research code/results retain their
original meaning. Costs and thresholds remain unvalidated research assumptions.

Typed API routes include health/readiness/model information, image/batch
prediction, triage configuration/simulation and experiment/metric discovery.
The dashboard has all eight requested modules. Simulation uses cached real
selection scores with three measured inference profiles; it rejects held-out
dataset caches and labels the outputs exploratory. Probability tables, CP sets,
decision reasons, risk/quality filters and CSV/JSON exports remain connected to
actual model outputs. Existing classification-card metrics remain bound to the
unchanged DR/quality/calibration artifacts, not to new triage claims.

The selected MobileNet and quality models remain deployed. EfficientNet-B0 is an
additional genuinely trained candidate, not an untrained architecture switch.
The local APTOS adapter verifies the actual source schema, images and connected
duplicates, without fabricating patient/eye/quality labels. The official APTOS
source is acquired, audited and configured at its relocated local root.

**Latest verification:** all 618 tests passed with genuine production/conformal
integration, zero failures/errors/skips, 130.82s; 25 targeted passed 11.29s. One upstream
AnyIO warning. Ruff/Compose/whitespace pass. Actual native and rebuilt Docker
JPEG/batch/TTA/readiness/conformal/simulator/overload workflows pass. All 70 runtime
source/configuration files match; PyArrow 25.0.1 is installed. The primary Docker
app runs on 18000/18501 after passing checks; the temporary stack was removed.
All 14 verified shards reuse without network, and three separately preselected
payloads pass bounded format/geometry checks without model/quality reassessment.
Evidence: reports/eyeq_hf_v1/verification.json. No new browser-rendering claim.

**Preserved APS/RAPS verification:** all 593 tests passed with genuine production/conformal
integration, zero failures/errors/skips, 109.14s; all 35 targeted tests passed in
12.89s. One upstream AnyIO deprecation warning. Same tested scientific/API/UI source
retained; Ruff/Compose/whitespace pass. Native and rebuilt CPU-container genuine
JPEG/batch/TTA/readiness/simulator/overload checks pass. Live-API AppTest verifies
20 records and all 40 adaptive alpha/seed/empty-policy/population combinations in
each runtime, 2,200 metric cells and 1,280 interval cells. All 68 runtime source/
config files match. Figure visually inspected; no new interactive browser-rendering
claim. Evidence: reports/adaptive_sets_v1/verification.json.

**Preserved EyeQ verification:** all 558 tests passed with genuine production/conformal
integration, zero failures/errors/skips, JUnit suite time 102.731s; all 22 targeted
tests passed in 11.840s. Tested source remains fixed. Ruff/Compose/whitespace pass.
Actual native and rebuilt CPU-container JPEG/batch/TTA/readiness/simulator/overload
checks pass, with matched original model card/sidecar. Live-API AppTest verifies
19 records, two source splits, all three quality categories and three nonzero table
disagreements. All 65 container source/config files match. No new interactive browser
rendering check. Current evidence: reports/eyeq_data_v1/verification.json.

**Preserved calibration verification:** 536 tests passed, no failures/skips,
100.90 seconds before the locked scalar/stress execution. All 48 clean and 48 stress
UI views and both figures were checked. The earlier native-only evidence is
preserved in reports/quality_calibration_stress_v1/verification.json.

**Preserved adaptation verification:** 522 tests passed, no failures/skips,110.17 seconds,
after the paired-seed metadata fix;521 passed128.16s before learning. Ruff and
Compose configuration pass. Actual native production JPEG/batch/TTA/readiness/
simulator/overload and live-API Streamlit checks pass:17 records, all six quality
runs and all12 seed/head controls, preserved original production card/sidecar.
All90 epochs/27 intervals and the metadata-only correction were independently
reconstructed. Both figures visually inspected; no new container/browser proof.
Current evidence: reports/quality_adaptation_v1/verification.json.

**Preserved APTOS verification:** 503 tests passed, no failures/skips, 91.16 seconds;
Ruff and Compose configuration passed. Actual native HTTP and Streamlit
component checks cover 16 records, new baseline/four-epoch/extended comparators,
both alpha/population controls and all 5 uniform seeds. Independent reconstruction
verified new source-development metrics and 134 preserved artifacts, including
consumed IDRiD evidence. Pre-execution 502 tests passed 82.77s; the 503-test suite
also covers the later owned-mask fix. Exact current evidence is
reports/aptos_optimization_v2/verification.json.
The earlier APTOS audit of 43,764 raw/fallback sets remains preserved.

**Preserved alignment verification:**198 tests passed, no failures/skips,78.51 seconds. This includes
real-model JPEG integration with the conformal sidecar. One upstream AnyIO
deprecation warning. An earlier full run exposed two obsolete assertions expecting
the old string error; the tests now assert structured overload codes/counts.
The final full run passed. A later one-line chart-default change passed its
targeted dashboard test (1 passed,4 deselected,8.87 seconds). Ruff and Compose checks
passed; the PowerShell launcher syntax parsed and now maps selected artifact paths.
The launcher itself was not used to claim deployment.

## Real experiments and their limits

| Candidate, same168-image selection set | Accuracy | Macro-F1 | QWK | Referable sensitivity |
| --- | ---: | ---: | ---: | ---: |
| MobileNetV3-small |63.10%|0.5372|0.7420|81.08%|
| ResNet18 |61.31%|0.5160|0.7443|75.68%|
| EfficientNet-B0 |64.29%|0.5249|0.7344|90.54%|

EfficientNet used 848 fitting images/212 patients, frozen official ImageNet features,
224px, seed 2026 and the declared settings. It completed 28 epochs/756 updates;
epoch 13 was selected. The macro-F1 rule retains MobileNet. Full source, input,
pretrained/checkpoint hashes and measured histories are recorded. Shared-workstation
training duration is not an architecture speed comparison. The comparison also
reports actual NLL/ECE/Brier/AUROC/AUPRC and checkpoint sizes; missing candidate
latencies are null. These are development results, not new confirmation.

Conformal fitting used independent84-image/21-patient temperature and84/21
score-calibration subgroups. Actual image checksums and patient/photo/fingerprint
disjointness passed. The sidecar applies its independent temperature to raw model
scores; displayed v1 grading probabilities remain unchanged. LAC and an explicitly
documented deterministic adaptation of the base paper's ordinal construction
were implemented independently; exact randomized reproduction is not claimed.

At alpha .10 on 168 selection images/42 patients:

| Method | Image coverage | Mean set size | Patient all-views coverage |
| --- | ---: | ---: | ---: |
| LAC |94.64%|3.345|85.71%|
| Deterministic ordinal greedy |100.00%|5.000|100.00%|

The ordinal quantile is1.0 under patient-maximum calibration, producing every
grade for every image. This is conservative but uninformative. Independent audit
reproduced every cached set in all three inference profiles. It found no credible
algorithm/integration bug. Small calibration groups, original stratification and
conditional TTA do not justify blanket finite-sample or shifted-data guarantees.

The fixed three-profile × fifteen-budget × eight-policy grid contains 360 cases:
288 overloaded and72 feasible. Every feasible case reviews100% of images and
has zero automatic clearance. No zero-miss benefit is inferred from full review.
These results fail to establish useful ordinal-guided automatic triage; no
thresholds were retuned after seeing them. Historical60% confirmation accuracy
still describes the unchanged classifier, not this policy.

## Docker and measured CPU workflow

The CPU image was rebuilt and the actual HTTP workflow passed real JPEG
single/batch/conditional-TTA, matched conformal/model-card, readiness/docs,
unsupported upload, simulator evaluated/overloaded and research endpoint checks.
The services use local ports 18000/18501; defaults 8000/8501 remain configurable.
The live browser rendered all eight modules and actual simulation overload.

A 12-image deterministic selection prefix, with two excluded warmups, measured
the complete loopback HTTP image-to-decision response:

| Profile | Mean request ms | P50 ms | P95 ms | Extra-pass fraction |
| --- | ---: | ---: | ---: | ---: |
| Single, first pass |304.64|264.89|425.56|0%|
| Single, extra pass |210.44|204.73|242.32|100%|
| Batch4, budget25% |1004.29|911.27|1262.66|25%|

The batch row is per four-image request (251.07ms/image amortized). Timings include
upload, JPEG decode, model/preprocessing, conformal, routing and JSON response;
client disk reads/browser rendering are excluded. Fixed profile order and normal
workstation variability prevent a speed-advantage claim; only three batch
requests were timed. API kernel peak RSS since container startup was 644,980 kB
(about 630 MiB); Docker/Desktop/dashboard overhead is additional. The measured
host has an i5-12500H, 16 logical CPUs, CPU-only PyTorch, and inference uses 2 threads.
Other laptop configurations and hard RAM limits have not been verified. Runtime
images exclude datasets; local data, environments and feature caches need
additional disk space. Source/environment and restart checks accompany deployment.

## Blockers, files and next work

**Current APTOS status:** download located and official competition source
confirmed on 3 November. [Safe local preparation](APTOS_SETUP.md) and the frozen
study protocol replace the earlier access blocker. IDRiD transport diagnostics
are completed and consumed; broader independent quality/prospective validation,
clinical review/recapture outcomes and prospective utility remain unproduced.

**IN PROGRESS overall.** Open work includes additional matched seeds, full
base-paper reproduction, official EyeQ photograph acquisition and terms, pixel
overlap and quality harmonization, independent validation, and useful triage
evidence. Future protocols will be locked before execution and use development
roles for tuning. Formal manuscript writing stays deferred.

Key files: MASTER_SPEC.md, CHANGELOG.md, AGENTS.md, TASKS.md, PROGRESS.md,
qsart/triage_v2.py, qsart/uncertainty/conformal.py, conformal sidecar, APTOS adapter,
EfficientNet training support, api/research.py, schemas/main, eight-page dashboard,
reproducible scripts, new tests and reports/spec_v1. Photos, patient manifests,
raw per-image scores, downloaded author code and checkpoints remain local/ignored.

The completed EyeQ extension adds qsart/data/eyeq.py, scripts/audit_eyeq_data.py,
configs/eyeq_data_v1.json, two test files, API/explorer support, setup/protocol/results
notes and eight aggregate/audit/native/container verification reports. The exact
master specification and historical experiment bytes remain preserved.

The fixed APS/RAPS extension adds qsart/uncertainty/adaptive.py,
qsart/evaluation_adaptive.py, scripts/run_adaptive_sets.py and plot_adaptive_sets.py,
config/protocol, two test files, API/explorer controls, nine JSON evidence reports,
PNG/SVG figure and results notes. Source photographs and raw individual results remain ignored.

The Hugging Face cache extension adds qsart/data/hf_eyepacs.py,
scripts/fetch_eyepacs_hf.py, configs/eyepacs_hf_v1.json, transport/join fixtures,
setup/plan/results notes and aggregate cache/native/container audit reports.
scripts/probe_aptos_384.py and its protocol/tests/two reports record the separate
fitting-only resource probe. All local dataset payloads/private IDs remain ignored.

**Current Docker verification, 6 November:** the PyArrow-pinned image was rebuilt
and exercised on temporary 28000/28501, then promoted to primary 18000/18501 after
the owned native services were stopped. Both primary services are healthy,
nonroot and read-only; all 70 source/config files match. The temporary stack was
removed. See reports/eyeq_hf_v1/primary_runtime.json and primary_deployment.json.

**Historical Docker recovery, 5 November:** the engine became available during this session
after an initial missing-pipe failure. Current source was actually rebuilt and
exercised in healthy, nonroot, read-only CPU containers on temporary ports
28000/28501; all 65 runtime source/configuration files match. This resolves the
host-startup blocker described in the historical paragraphs below. The native app
remains on 18000/18501. Container evidence: reports/eyeq_data_v1/container_runtime.json.

Evidence: [base-paper critical study](research/BASE_PAPER_STUDY.md),
[declared alignment protocol](research/SPEC_V1_ALIGNMENT_PROTOCOL.md),
[real development report](../reports/spec_v1/development_evaluation.json),
[model comparison](../reports/spec_v1/model_comparison.json),
[deployment](../reports/spec_v1/deployment.json),
[CPU report](../reports/spec_v1/end_to_end_cpu.json).


Final deployment checks: all **37 runtime Python files** match container/host
hashes. Actual JPEG/conformal/batch/simulator/research checks passed again after
API restart; see reports/spec_v1/deployment_after_restart.json.

The Docker paragraph above records 21 October execution. On 22 October the host
engine failed to start because of an inaccessible stale socket. Automatic approval
review blocked socket removal; no factory reset was attempted. Native services
were restored using the existing PowerShell launcher on ports 18000/18501, with
real-image/model/conformal/batch/simulator HTTP checks. A new container run was
**BLOCKED** at that checkpoint; current 5 November container proof appears above.
Historical Docker evidence is preserved.

22 October verification: **223 passed**, no failures/skips, 67.03 seconds with
genuine-model/sidecar integration enabled; Ruff and Compose configuration passed.
The added real development experiment has exact clean-equivalence checks on all
168 images. Independent reconstruction verified all 1,176 LAC sets, all seven
conditions' metrics, six paired accuracy intervals and the recorded hash bindings.
The native HTTP workflow passed again after restarting the launcher, and the new
experiment was retrieved through the actual research endpoint. Evidence is in
reports/robustness_v1; the previously measured 198-test/container results remain
historical records rather than being overwritten.

Further 22 October verification: **299 passed**, no failures/skips, 59.59 seconds,
with genuine-model/sidecar integration enabled; Ruff and Compose config passed.
The intermediate pre-registry full run was 297 passed, 72.75 seconds. Both new
experiments are exposed through the real API after a native launcher restart;
the original model card and sidecar remained matched. Docker startup was
unavailable then. At that historical date APTOS was not yet acquired.

The ordinal diagnostic reproduces the original four baseline fits exactly and
explains the threshold-one atom with three images from two patients. At alpha .10,
five randomized patient-max runs average size 3.644, image coverage 96.07% and
patient all-views coverage 86.19% (range 80.95–92.86%). These are randomization
replicates, not training seeds or formal coverage guarantees. Only 2–7 images
receive guarded pointwise automatic recommendations; no budget utility is claimed.
All 28 comparisons and independent 4,704-set reconstruction are retained.

The new head initialization rule is independent of feature-cache existence.
All six real DR/quality runs use seeds 2026/2027/2028, one source-valid new cache,
848 fitting and 168 selection images; calibration/confirmation were unused.
DR macro-F1 is .53009 ±.00782 and overall-quality accuracy 76.79% ±.60 percentage
points (sample SD). Defect heads remain weak, including field/artifact means below
training-majority constants. No favorable seed was selected or deployed. Exact
artifact hashes distinguish these runs from historical version-string collisions.

The independently tuned guarded-policy study is **VERIFIED COMPLETE** as a
development experiment. Protocol commit `4167118` preceded execution source
`c2f8fbb50622631c6001092f7b1da008498f421b`. Three patient folds independently tune
six decision prices for eight policies under three uncertainty modes and four
budget pairs, while all outcomes use the same fixed evaluation ledger. There are
1,728 candidate allocations and 192 tuned/default scenarios: 78 fully evaluated,
114 incomplete with null pooled outcomes. All 19 comparisons with matching actual
per-fold workloads have identical actions; the other 54 estimable pairs have
unequal workloads. The deployed ordinal gate permits no automatic clearance.
Independent reconstruction checked all candidate choices, 576 fold records,
capacities, paired outcomes and 219 intervals. Full
[results](research/POLICY_COMPARISON_V1_RESULTS.md) preserve unfavorable tuning
outcomes. These previously inspected selection patients do not provide fresh
end-to-end validation; no policy or artifact was selected for deployment.

Final comparator integration verification: **371 passed**, zero failures/skips,
61.26 seconds with genuine-model/conformal integration enabled; Ruff and Compose
configuration passed. Actual native HTTP checks and Streamlit AppTest against the
running API verified all ten experiment records, the new192-scenario report,
null incomplete outcomes and diagnostic labels. A nested-table serialization bug
found by that integration was fixed with a regression fixture. Browser observation
was rejected by its URL policy, so this session has no new browser-rendering proof.
Native checks did not replace a new container run; Docker was blocked at that checkpoint.

The APTOS study uses 3,514 images after quarantining 148 images in 51 conflicting
fingerprint components without relabeling. Training/selection/calibration/reserved
test counts are 2,115/521/527/351. Temperature fitting uses 262 separate images;
conformal scores use 254 representatives from other calibration components.
EfficientNet-B0 wins macro-F1 with .60334 versus MobileNet .57019, at 78.69% versus
76.97% accuracy. MobileNet has higher QWK; no universal architecture advantage is
claimed. EfficientNet grade-3 recall is only 6/26. Temperature 1.89805 reduces
selection NLL .68315 to .58069 and ECE10 .09458 to .04319.

LAC alpha .10 selection coverage is 86.37%, with grade-3 coverage 42.31%.
Randomized ordinal raw mean coverage is 91.36% with mean size 1.791; the common
argmax fallback raises these to 94.82% and 1.825. Grade-4 coverage remains only
67.5–75% across seeds. All raw/fallback, per-grade and representative outcomes are
retained. These results are development diagnostics on selection data and unknown
patient relationships, with label-dependent quarantine; they establish no patient,
conditional or external coverage guarantee. All 14,588 raw/fallback sets were
independently reconstructed. Selected production artifacts remain byte-identical.

Earlier 4 November checks:405 tests passed in98.87s with real models/conformal;
Ruff and Compose configuration passed. Actual native JPEG/batch/TTA/simulator,
APTOS research routes and live-API Streamlit component checks passed. Relocated
source annotation and all3,662 PNGs match original hashes. Inventory now honors
the ignored local source setting with precedence tests. Docker remained
unavailable at that checkpoint; new container/browser checks were unverified then.
