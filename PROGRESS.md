# Execution log

## 2026-11-04 — completed reserved IDRiD integrity/overlap audit

- Executed from clean d597daa553b58c57f8e96f1b33a2f24bf8a9e4f8. All 516
  disease-grading JPEGs decoded and were bound by source SHA256/pHash/dimensions.
  Original 413/103 partitions remain external_holdout; every row has training
  eligibility False. Original DR 0–4 and separate DME-risk 0–2 labels are retained.
  No classifier inference, model fitting, calibration or external outcome exists.
- Fixed pHash distance <=2 found 8 internal pairs, 6 byte-exact pairs, 7 duplicate
  components and 4 label-conflicting components. Against 5,186 source fingerprints,
  three perceptual candidates link three IDRiD images to APTOS; none is byte-exact.
  No DeepDRiD matches were detected. All observations/labels remain unchanged;
  an independent-cohort claim requires candidate adjudication and source clarity.
- Independent scipy-DCT reconstruction decoded/rehashed all 516 images and exactly
  reproduced all 132,870 internal/2,675,976 cross-source distances, candidates,
  components, source-label mappings and aggregate counts. All 20 preserved prior
  master/production/APTOS artifacts matched. Aggregate report SHA256:
  b31891fc14c5c52ab1e638821a5e014762e20d2add6a936ab18d093ef165f88a.
- Full 456-test suite passed 75.91s, zero failures/skips, with real production-model
  and conformal integration; Ruff/Compose passed. Actual native JPEG single/batch/
  TTA/readiness/simulator and live-API Streamlit checks passed after restart.
  Thirteen evidence records include IDRiD reserved-data scope, source/overlap
  warnings, original-partition counts and zero external predictions. Production
  card/sidecar remain matched. No new browser or container execution is claimed.
- Download-origin question remains unanswered: report records unconfirmed local
  package origin, without treating license/layout as archive authentication.
  Docker engine remains unavailable. Formal manuscript stays deferred. Next:
  resolve provenance and adjudicate observed candidates under a fixed rule,
  then declare locked external evaluation; stronger development optimization
  remains separate. Overall scientific acceptance remains IN PROGRESS.

## 2026-11-04 — declared IDRiD external-data readiness audit

- Found the local disease-grading release while checking remaining evidence gaps.
  Official documentation describes 516 images / 413 original training + 103 test.
  Only source headers/counts and license were inspected before the declared
  protocol; no external model inference, fitting or tuning was performed.
- Added strict original-layout parsing, separate DR/DME fields, full JPEG integrity,
  fixed-distance fingerprint overlap and external-only training-rejection guards.
  The source-confirmation question remains separate from local file/layout checks.
- Generated software fixtures:21 tests passed,5.85s; targeted Ruff passed. These
  are plumbing tests, not evidence of external grading performance. Added a
  dedicated data-readiness explorer view with explicit reserved/not-inferred text.
  Real audit will execute only from clean committed source, with frozen references.
- Full final pre-execution suite:456 passed, zero failures/skips,75.91s with real
  production-model/conformal integration enabled. Ruff and Compose configuration
  passed. An initial new API assertion confused discovery metadata with the
  narrower metrics schema; corrected the test and reran the complete suite.

## 2026-11-04 — completed bounded APTOS fine-tuning and audit

- Real execution used clean source 8357e9d918ef9ec00d1f66e4efadc3325790097c. The pre-optimizer
  canonical-cache equivalence check matched all 521 original predictions; maximum
  probability delta 6.00379896e-06, below declared 1e-5.
  All 3 seeds completed 4 epochs / 532 updates each from the SAME selected warm start.
  All 211 backbone parameter tensors changed; all 147 BatchNorm buffers stayed fixed.
- Seed 2026: selected epoch 4, accuracy 0.78310940, macro-F1 0.63348889, QWK 0.86005154; grade-3 recall 0.65384615, grade-4 recall 0.47500000.
- Seed 2027: selected epoch 4, accuracy 0.78502879, macro-F1 0.62077703, QWK 0.85664515; grade-3 recall 0.30769231, grade-4 recall 0.70000000.
- Seed 2028: selected epoch 4, accuracy 0.77543186, macro-F1 0.61807210, QWK 0.85668708; grade-3 recall 0.38461538, grade-4 recall 0.55000000.
- Conditional macro-F1 mean 0.62411267 / sample SD 0.00823190.
  Primary seed 2026 remains fixed, including its severe-grade tradeoff; no best seed
  or alpha was promoted. Every run's descriptive paired interval is retained.
- Independently fitted 3 temperatures using 262 role images each; separate 254 CP
  representatives support all 42 declared comparisons. Audit reconstructed all
  43,764 raw/fallback sets, every epoch metric/selection, schedules/flip counts,
  six paired intervals, all 3,163 development image bytes and both canonical caches.
  Original quarantine/manifests/roles, baseline reports, master and production
  artifacts remain byte-identical. No reserved-test inference occurred.
- Actual native research HTTP equality and Streamlit AppTest passed: 12 experiments,
  baseline + 3 run rows, 42 comparisons, all 5 uniform seeds and training-seed/alpha/population
  controls. Original production card/sidecar matched. Scientific figures were
  generated and visually inspected; new browser rendering is not claimed.
- Full pre-execution suite: 433 passed, zero failures/skips, 75.66s; Ruff and Compose
  configuration passed. Current source is the tested committed execution source;
  post-run additions are aggregate reports/figures/documentation; one plot label
  was shortened after visual QA without changing measured bytes. Exact commands,
  JUnit/report/audit hashes and deployment evidence are in verification.json.
- Four CPU epochs and a common selected initialization limit conclusions; this
  establishes neither convergence nor exact-paper optimization, independent-test
  performance or clinical utility. Overall acceptance remains IN PROGRESS. Docker
  engine remains unavailable; stronger optimization, external evaluation and useful
  validated triage remain open. Formal manuscript work stays deferred.

## 2026-11-04 — declared APTOS parameter fine-tuning follow-up

- Started from clean published60a7e81 and preserved all completed studies. The
  new study tests supervised backbone adaptation after the frozen-feature study's
  weak severe-grade findings; selection data are explicitly already inspected.
- Declared protocol/config at ef2813ff86b0cf8ce1b5c6a523eaf286284ac23d before any
  real fine-tuning. Generated-tensor capacity probes measured roughly2.2s/update
  for contiguous/two-thread execution and .8–1.35s with channels-last/four threads.
  Before real learning, revised12epochs to a bounded4epoch/three-seed pilot,
  four threads/channels-last and cosineTmax4. This is not convergence evidence,
  a controlled speed claim or exact base-paper optimization reproduction.
- Seeds2026/2027/2028 share the same verified APTOS B0 warm start. Every learnable
  parameter adapts while BatchNorm running buffers remain fixed. The primary
  seed2026 is predeclared regardless of outcomes. New canonicaluint8 input cache
  avoids repeated high-resolution geometry, with normalization equality and
  cache-hit/miss RNG regression coverage. Runtime paths honor the relocated root;
  original manifests/quarantine/roles and reserved test remain preserved.
- Separate ownership under master26 covers core training/cache/resume, execution
  and figures, independent verification, and root API/UI/documentation integration.
  Added aggregate discovery and display-only training-seed controls; changing
  that control cannot select/deploy a winning seed. Root API/UI subset23passed,
  53.16s; Ruff passed. No real fine-tuning outcomes are claimed by these fixtures.
- Final pre-execution verification:433passed,zero failures/skips,75.66s with
  genuine production DR/quality/CP artifacts enabled; Ruff passed. Actual default
  readiness validated the relocated runtime cohort and frozen bindings without
  inference/training. Independent source review fixed a stale-best checkpoint
  resume window; interruption, changed-config and stale-selection regressions
  now pass. Full execution is launched only after this source is committed clean.

## 2026-11-04 — APTOS completion, relocation and application integration

- The run from frozen e862700331186e6a754148b33534974d0f716141 completed both
  genuine frozen-ImageNet-feature candidates and all14 CP comparisons in1435.90s.
  EfficientNet:36epochs,2412updates,selectedepoch21,selection78.6948%accuracy,
  .603341macro-F1,.844005QWK. MobileNet:20epochs,1340updates,selectedepoch5,
  76.9674%accuracy,.570192macro-F1,.855747QWK. The fixed F1 rule selects B0;
  it does not win every metric. Both backbone states remain unchanged.
- Independent temperature1.898052256 uses262 images. Separate254 conformal
  representatives fit scores. Selection NLL .683146→.580690 and ECE10
  .094580→.043193. LACalpha.10 coverage86.37%/size1.296; grade3coverage42.31%.
  Randomizedalpha.10 raw mean coverage91.36%/size1.791, fallback94.82%/1.825;
  grade4coverage67.5–75% persists even with fallback. Classifiergrade3recall6/26
  andgrade4recall22/40 remain weak. No new model/uncertainty seed is deployed.
- Final independent audit passed:original archive/all3,662image hashes, complete
  grouping/partitions/roles, both histories/checkpoint choices, official frozen
  states, direct head×feature probabilities and all14,588raw/fallback sets.
  Max probability error1.90e-7,quantile error1.17e-8; independent temperature
  optimum NLL gap2.66e-11. Source/privacy/master/production-byte checks passed.
  Evaluation SHA994a8e78148d6c16d601d61d8774e0d36070b653ebf5549dec093ecbad4924ec.
- User relocated/extracted the source into the major-project folder. Verified the
  new train.csv and all3,662trainingPNG SHA values against frozen source evidence;
  no model/test inference or split changes. Configured QSART_APTOS_ROOT in the
  ignored .env. The inventory CLI now reads this local setting, with environment
  and explicit-argument precedence and project-relative resolution. Four targeted
  inventory tests passed6.72s. Original manifests/cache remain untouched.
- Full frozen pre-relocation suite passed402tests,85.41s, no failures/skips,
  real-model/conformal integration enabled. Later location-reader regressions
  require the final updated suite recorded below. Plot-only label wrapping was
  corrected after visual QA; both scientific figures were generated and inspected.
- Native application was no longer listening after the interruption. Restarted
  the hidden launcher on free18000/18501. Actual APTOS HTTP discovery/metric
  equality and Streamlit AppTest passed:11experiments,2candidates,14comparisons,
  both selection populations,all5randomization seeds,raw/fallback distinctions,
  matched original model card/sidecar. This is component verification, not a new
  browser-rendering claim. The relocated source is detected by the real inspector.
- Docker info still reports the missing engine pipe on4November. No destructive
  repair was attempted. Formal manuscript work remains deferred; full fine-tuning,
  multiple classifier seeds, independent external evaluation and useful validated
  triage remain open. No fresh test performance or quality supervision is invented.
- Final updated full suite:405passed, zero failures/skips,98.87s with genuinely
  trained checkpoints/conformal integration enabled. Ruff, Compose configuration
  and Git whitespace checks passed. Actual JPEG single/batch/TTA/readiness,
  matched card/sidecar, unsupported upload and simulator evaluated/overloaded
  checks passed again after restart. New source setting tests verify project-
  relative resolution, environment precedence and preserved original fallback.
  Exact machine-readable evidence is reports/aptos_v1/verification.json.

## 2026-11-03 — official APTOS archive and declared development study

- Resumed clean published996deb6 and inspected current instructions/evidence.
  User reported a downloaded dataset; located the canonical competition ZIP
  locally,10,215,289,875bytes and3,662training-image members. User explicitly
  confirmed the official Kaggle competition source. No further upload is needed.
- Committed the new protocol/config before real model training at
  a7f59b97fe3bde3f074e47c0a47c97626747d6a5. Later pre-execution clarifications
  state the inherited LR scheduler and initial source-label access for grouping;
  reserved test INFERENCE remains prohibited. The protocol is distinct from
  the paper's384px/full-fine-tuning experiment and does not claim reproduction.
- Used the master specification's section26 parallel workstreams with separate
  ownership: archive/data validation; model/calibration runner and figures;
  independent read-only verification; root API/backlog/deployment integration.
- Implemented safe training-only archive extraction, original CRC/SHA/decode
  validation and explicit whole-conflicting-component quarantine. Source labels
  are preserved; default strict rejection remains. Kept manifests, original-image
  identifiers, exclusion records, score arrays and eventual weights ignored locally.
- Prepared fixed60/15/15/10fingerprint-group splits, a test-free development
  manifest, independent calibration roles and label-independent representatives.
  Unknown patient/eye identity is reported; no APTOS quality labels are invented.
- Implemented two frozen official ImageNet candidate runs and all14 declared
  conventional/deterministic/randomized conformal comparisons, raw empties and
  common modal fallback. Current production DR/quality/card/sidecar SHA values
  were snapshotted before execution. No new artifact is selected for deployment.
- Added APTOS aggregate discovery with the correct dataset/evidence scope.
  API regression suite passed14tests in31.35s; targeted Ruff passed. The first
  sandboxed API test attempts hung in Windows socketpair/AnyIO startup; the
  timeout traceback established denied loopback. Repeating with permitted local
  loopback passed; the identified blocked test processes were stopped.
- On3November, Docker info could not reach the DockerDesktopLinuxEngine pipe.
  A missing server pipe establishes unavailability, not a new confirmed cause.
  Historical container results remain separate from current verification.
- Requested a non-destructive Docker Desktop start; the CLI accepted it, but the
  engine never became reachable. Current3November backend logs confirm startup
  failed while renaming the stale sailor-ingest socket, which Windows could not
  access. No deletion, factory reset or further repair was attempted.
- Final pre-execution code checks:401tests passed, zero failures/skips,103.05s
  with genuine DeepDRiD DR/quality/conformal integration enabled. The command was
  `.venv/Scripts/python.exe -m pytest -q --junitxml=reports/aptos_v1_tests.xml`.
  One upstream AnyIO warning remains. Ruff and Compose configuration passed.
  Data/archive subset:27passed,18.89s; research-runner subset:12passed,51.75s.
- Actual data preparation and subsequent training outcomes are recorded below
  after execution; no result is inferred from implementation or fixture tests.
- Real data preparation completed in856.26s. All3,662 source training PNGs passed
  decode and extracted-byte hash checks. The fixed fingerprint graph found255
  candidate pairs,134 byte-exact pairs and51 label-conflicting components with
  148images. Entire conflicting components were quarantined without relabeling;
  this exclusion changes the study population and may include false candidates.
- Retained3,514images/3,399components. Fixed partitions:train2,115/2,048groups,
  selection521/507,calibration527/507,reserved internal test351/337. Temperature
  role262images/253groups; conformal role265/254,with254representatives used for
  score-threshold calibration. Development manifest has3,163non-test rows andSHA
  681799c7ee5f04e988dc607f6cf296fd882c5f55f2603dcd0a2ccd8148951fab.
  Source train.csv SHA b2a3479695922e62c83c94a1a9ef669b4b6d68cc26a1d7090ebedecdd7dd79a5.
  Exact manifests, group membership and exclusions stay ignored; public data
  audit is aggregate-only. No competition test was extracted and no internal
  reserved-test model inference was performed.
- Independent data reconstruction verified the entire source fingerprint graph,
  every whole-component exclusion, all partitions/representatives/roles and all
  3,662 original image byte hashes. Actual readiness found a pandas CSV column
  ordering comparison defect before training. Explicit column ordering fixed it;
  a regression covers reordered full-manifest columns without accessing reserved
  grades for model preparation. Updated runner subset:13passed,11.41s; Ruff passed.
  Actual readiness then passed with the frozen manifests and no model inference.

## 2026-10-22 — independently tuned guarded-policy comparison

- Verified prior published e23a18b and successful PR workflow39822233255.
  Declared POLICY_COMPARISON_V1_PROTOCOL.md at4167118 before implementation and
  execution. Frozen execution source is c2f8fbb50622631c6001092f7b1da008498f421b.
- Added research-only fixed-ledger evaluation, explicit ordinal/LAC/no-CP modes,
  complete patient folds, independent six-price tuning and overload accounting.
  Allocator prices cannot change outcome evaluation prices. Production triage,
  selected weights, independent temperatures and sidecar bytes are unchanged.
  New helper/runner plus existing triage regressions:94 passed,8.57s; Ruff passed.
- Ran `.venv/Scripts/python.exe scripts/run_policy_comparison.py` successfully in
  19.38s. Reused frozen genuine first-pass scores for168 selection images/42
  patients and verified all original image bytes/source/input/artifact bindings.
  No new model/calibration/confirmation/test/external predictions were generated.
  Three folds each exclude14 patients/56 images and tune on28/112. These patients
  previously selected the classifier; this is development policy cross-fitting.
- Retained all1,728 candidates (786 evaluated,942 overloaded),192 tuned/default
  scenarios (78 complete,114 incomplete with null pooled metrics),576 fold records
  (256 evaluated,163 held overloads,157 skipped untunable). Untunable diagnostics
  describe tuning N112, not an attempted held-fold allocation. Pooled capacities
  sum actual fold floors. No unsafe forced clearance or failed-fold exclusion.
- Independent reconstruction verified all allocations, all131 selected prices,
  fixed ledgers, exact caps, folds,180 planned comparisons/73 estimable pairs and
  all219 descriptive bootstrap intervals. All19 actual-workload-matched pairs
  have identical individual actions and zero pointwise outcome deltas. The other
  54 estimable pairs have unequal workloads. No resource-matched advantage found.
- Ordinal production mode permits zero automatic cases; only full review is
  feasible. LAC dual tuning at B1/.4 raises fixed proxy .25 to .316369, with two
  automatic cases and two severe recaptures. No-CP B.6/.4 tuning clears18/168,
  with one referable automatic miss and five severe recaptures. Negative outcomes
  are preserved; no method, uncertainty mode or price was promoted to deployment.
- Added SHA-bound compact summary and192-cell feasibility/coverage figure via
  scripts/summarize_policy_comparison.py (14 tests passed). Full measured JSON
  SHA4dfd99d535d7f7413596caab5ca1fa47629a62126a488b907563656bf5834085
  stays unchanged. Public reports contain aggregates only; private fold/actions,
  original execution plot and reproduction scripts remain local under artifacts.
- Initial full genuine-model/conformal suite:371 passed,64.60s, one upstream
  AnyIO warning. Later real-API Streamlit integration exposed heterogeneous
  nested arrays in the experiment inventory table; scalar-only summaries fix
  Arrow serialization while full detail views retain all comparisons. Added a
  regression fixture. Final post-fix verification is recorded below.
- Final full post-fix suite with all six genuine artifact integration environment
  settings:371 passed, zero failures/skips,61.26s, one upstream AnyIO warning.
  Command: `.venv/Scripts/python.exe -m pytest -q
  --junitxml=reports/policy_comparison_v1_tests.xml`. Final Ruff, Compose config and
  Git whitespace checks passed. API/dashboard targeted subset also passed19 tests
  in42.41s before the later inventory-table fix; final full run includes that fix.
- Native launcher restarted on18000/18501. First verifier attempt preceded API
  readiness and was refused; after actual startup, real JPEG single/batch/TTA,
  card/sidecar matching, readiness/docs, unsupported upload, evaluated/overloaded
  simulator and dashboard HTTP health passed. Actual research endpoint returns
  all192 scenarios,180 comparisons,114 null outcomes and exact bootstrap settings.
  Streamlit AppTest against the running API loaded all10 experiments, verified
  ordinal24 incomplete/null plus8 full-review tuned rows and ablation notes.
- Browser URL access policy rejected observation; no workaround or new browser
  screenshot claim. UI evidence is explicitly Streamlit component testing, not
  browser rendering. Current Docker startup remains unavailable; prior container
  evidence is preserved. APTOS is still not acquired, IDRiD unassessed, useful
  validated triage unproven, and formal manuscript work remains deferred.

## 2026-10-22 — ordinal randomization and repeated training seeds

- Verified prior published commit4659374: GitHub PR workflow39814919309 completed
  successfully. Declared both new protocols in commit99a47b5 before new outputs.
- Added explicit-uniform randomized ordinal research scores, separate from the
  unchanged production artifact/schema. Mathematical tests cover candidate-score
  ordering, ties, contiguous inversion, exact ranks, grouped maxima, empty fallback,
  alpha nesting and safe artifact/provenance handling. New and existing conformal/
  transfer regression subset: 120 passed, 12.96s. Ruff passed.
- Diagnosed and fixed cache-dependent classifier-head RNG. The feature extractor's
  DataLoader iterator advances Torch RNG on cache misses; head construction now
  reseeds immediately afterward with the declared training seed. Meaningful fixture
  cache-miss/hit DR+four-head quality states/history/probabilities match exactly;
  different seeds remain distinct. Existing source/cache resume guards remain.
- Executed scripts/run_ordinal_study.py at16bd809 on 84 original calibration images
  /21 independently reserved patients, with unchanged independent temperature;
  reused bound clean raw scores for 168 selection images/42 patients. Verified 252
  actual original image hashes and old model/calibration/cache/split bindings.
- Reproduced four original LAC/deterministic ordinal fits and selection sets with
  zero threshold error. Three terminal true-label images from two patients give
  two patient maxima at 1, forcing both ordinal rank20/rank21 thresholds to 1.
- All 28 declared method/unit/alpha/seed rows retained. At alpha .10 randomized
  patient-max mean size 3.644, image coverage 96.07%, patient-all-views 86.19%
  (range 80.95–92.86). At alpha .05: size 4.029, image 98.45%, patient 94.29%.
  Randomization seeds 2026–2030 are not model-training replicates. Pooled-image
  diagnostic is narrower but changes the target and contains correlated views.
  Guarded primary alpha .10 automatic counts are 6/2/7/5/7 in seed order,
  range 2–7 of 168; no automatic true-referable case, with no safety/utility claim.
  Original deployed deterministic sidecar stays unchanged.
- Independent audit without project scoring code reconstructed 28 fits and 4,704
  sets plus raw sets, all metrics/actions, all seed summaries and 252 image hashes.
  Maximum threshold error 4.44e-16. Source/protocol/cache/package/privacy bindings
  passed. Public audit is in reports/ordinal_study_v1.
- Committed ordinal aggregates and froze clean training source99a6cbbce0204c965c8221d237cf7cf644e57ddf.
  Ran scripts/run_repeated_seeds.py --execute --protocol
  docs/research/REPEATED_SEEDS_V1_PROTOCOL.md --source-commit
  99a6cbbce0204c965c8221d237cf7cf644e57ddf. All 6 DR/quality runs at seeds
  2026/2027/2028 completed; every provenance dirty flag is false. Fresh MobileNet
  extraction 119.08s, then five source-valid cache hits with identical official
  weights/features. All 1016 training
  and selection image bytes were checked. Calibration/confirmation were unused.
- DR seeds: selected epochs 15/24/25, executed epochs 30/39/40, updates 810/1053/1080;
  macro-F1 .536774/.521493/.531992, mean .5300866, sample SD .0078167.
  Accuracy 62.8968% ±.3437 percentage points. Overall quality selected epochs 4/10/7,
  executed 19/25/22, updates 513/675/594; accuracy mean 76.7857% ±.5952 points.
  Defect clarity/field/artifact means 48.81/51.59/41.87%, sample SD 5.29/6.56/9.68 points.
  Field/artifact fall below post-run audit training-majority constants 69.05/47.62%.
  All seed outcomes retained; historical seed2026 excluded from the new procedure mean.
- Independent repeated-seed audit checked all 6 sources/configs/caches/weights,
  prediction rows/labels, all metrics/supports and 21 means/sample SDs, epoch/update
  counts and 48 run-artifact hashes. Direct saved-head/frozen-feature probabilities
  agree within 1.63e-7; best/last backbone tensors match official weights. Original
  selected DR/Q/basecalibration/card/conformal hashes are unchanged. These checks
  use saved evidence, not OS-level I/O tracing or a new retinal inference run.
- Full real-model/conformal suite before registry update: 297 passed, 72.75s.
  Final suite after both aggregate endpoints: 299 passed, 59.59s, zero failures/skips,
  one upstream AnyIO warning. Ruff and Compose config passed. Exact JUnit and
  public summary are preserved separately from prior test evidence.
- Restarted native PowerShell launcher; actual real JPEG single/batch/TTA,
  card/sidecar matching, readiness/docs, unsupported uploads, simulator evaluated/
  overloaded and dashboard health all passed. HTTP research discovery returned
  all 28 ordinal comparisons and all 6 training runs. No new Docker execution is claimed.
- Browser verification displayed nine experiment records, selected repeated_seeds_v1
  and rendered its actual three-seed mean/SD. Saved aggregate-only explorer proof.
- Remaining: independently tuned policies and useful validated triage, official
  APTOS acquisition/full reproduction, external IDRiD and additional robustness.
  Overall research remains IN PROGRESS; formal manuscript work stays deferred.

## 2026-10-22 — controlled degradation and publication

- Published specification alignment as commit 20009517f79eefc1c7bd2f34474781de7ad10f79
  on the existing draft PR #1. GitHub push and pull-request verification both
  succeeded; PR run 39812507691 passed install, Ruff, pytest and Compose checks.
  Main is unchanged; photos, manifests, individual scores and weights stay local.
- Declared ROBUSTNESS_V1_PROTOCOL.md at commit 8c0c83b before any new condition
  scores. Research-only canonical-input helper and exact decimal conformal-rank
  fix are at execution commit 493ad7c6e37d16dc338e00280d8ac714ee23a0c8.
  The rank fix prevents an extra conservative rank at n=9/alpha=.7; current
  n=21 alpha .10/.05 sidecar and numerical outputs remain unchanged. Image/grouped
  boundary tests plus conformal sidecar tests: 43 passed, 7.75s.
- Ran `.venv/Scripts/python.exe scripts/run_controlled_degradation.py` on the
  complete frozen selection population: 168 real images / 42 patients, seven
  conditions, 1,176 predictions. Canonical crop/resize occurs once, followed by
  identity, blur radii 1/2, intensity factors .5/1.5 and JPEG qualities 50/20.
  Clean DR/quality/three-defect/raw-DR probabilities matched ordinary inference
  exactly on all 168 images. Original SHA bytes and all selected artifact hashes
  were verified. No training, recalibration, gate changes, TTA or condition tuning.
- Actual DR accuracy: clean 63.10%, blur1 48.21%, blur2 34.52%, dark 59.52%,
  bright 58.33%, JPEG50 44.64%, JPEG20 22.02%. JPEG20 paired accuracy change is
  -41.07 percentage points, descriptive patient-bootstrap interval [-53.59,-26.79].
  Mean poor-quality score is .541 clean and .511 JPEG20; its difference interval
  crosses zero, so no statistically established decrease is asserted. Its failure
  to rise with compression damage exposes a model-response weakness. Blur raises
  referable sensitivity while reducing specificity; no benefit is claimed.
- Fixed LAC alpha .10 image coverage falls from 94.64% to 73.21% for JPEG20;
  all-views coverage falls from 85.71% to 45.24%. All ordinal sets remain all five
  grades, and all pointwise actions recommend review. No shifted coverage,
  automatic-triage utility, capacity feasibility or successful care is established.
  Natural-quality labels are context strata; only clean quality accuracy (74.40%)
  is assessed against them. Original disease labels remain underlying-disease
  labels even where corruption hides visible evidence.
- Private per-image caches/hashes remain under artifacts/robustness_v1. Public
  aggregates, three plots and study notes are under reports/robustness_v1 and
  docs/research/ROBUSTNESS_V1_RESULTS.md. Independent audit reconstructed all
  1,176 LAC sets, seven conditions' metrics, six paired accuracy bootstrap intervals,
  first/last canonical pixels and all 14 corresponding transformed-image hashes;
  input/artifact/cache/protocol/nine-source bindings and privacy checks passed.
- Added aggregate robustness discovery to /research/experiments and /research/metrics
  for the existing dashboard explorer. Fixture contract test preserves an unfavorable
  result and development-only scope. Targeted endpoint tests: 10 passed, 14.08s.
- Full verification command used QSART_REAL_MANIFEST=development_v1.csv,
  QSART_REAL_SPLIT=validation, genuine selected DR/quality/calibration and
  QSART_REAL_CONFORMAL=conformal_r1.json, then `python -m pytest -q
  --junitxml=reports/robustness_v1_tests.xml`: **223 passed**, no failures/skips,
  67.03s, one upstream AnyIO deprecation warning. Ruff and Compose config passed.
- Docker Desktop startup failed before the Linux engine: inaccessible stale
  sailor-ingest socket. Rename failed; automatic approval review rejected socket
  removal. No factory reset or data deletion occurred. Current Docker execution
  remains blocked; historical Oct21 container proof stays unchanged.
- Started the existing PowerShell local launcher in the background, then restarted
  it after API integration. Real JPEG single/batch/TTA, readiness, matched model
  card/conformal, unsupported uploads, simulator evaluated/overloaded and dashboard
  health checks passed at 18000/18501. scripts/verify_deployment.py now records the
  invoker-established runtime explicitly; current evidence is native_python, not
  a new container run. The real research endpoint returned all seven conditions.
  Browser verification selected controlled_degradation_v1 in Experiment Explorer
  and displayed development_model_input_stress, 168 images and 42 patients;
  aggregate-only screenshot is saved as explorer_preview.png.
- APTOS is still not downloaded, and IDRiD is not evaluated. Useful ordinal sets,
  independently tuned policies, repeated seeds and external evidence remain open.
  Formal manuscript drafting remains deferred. Overall research is IN PROGRESS.

## 2026-10-18 — audit and bootstrap

- Inspected parent directory and all starter ZIP member names; read starter
  README, data parser, policy, CLI, tests and research protocol.
- Queried GitHub repository inventory and target contents through the connected
  GitHub plugin; the user selected Riteshhai/Retina-Opthalmology.
- Preserved originals; extracted starter to QSART; preserved full build prompt.
- Python 3.11 detected; torch/torchvision absent initially. No project retinal
  images or trained checkpoint found at initial audit.
- Parallel work: dataset ingestion; model/training/calibration; source and
  literature verification. Root integrates policy, API, dashboard and packaging.
- Real-image training and deployment are not yet verified.

## 2026-10-18 — genuine pilot and application verification

- Downloaded official DeepDRiD metadata, annotation definitions and license at
  commit 56d8af71319803150d0e7b90533e0918c7bf85ee. Acquired 260 JPEGs
  (138.9 MiB), representing 65 patients. Strict audit: all readable, no missing
  or quarantined pilot images, no cross-split SHA or texture pHash candidates.
  Eleven ambiguous rows in the full annotations are quarantined.
- Frozen manifest SHA256:
  e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0.
  Patient-disjoint partitions: 160 training, 20 checkpoint-selection validation,
  20 calibration and 60 holdout images. See docs/DATA_STATUS.md.
- Installed torch 2.14.1+cpu and torchvision 0.29.1+cpu in the local .venv.
- Executed scripts/train_dr.py and train_quality.py with 3 epochs, 96-pixel input,
  batch size 8, seed 42 and 2 CPU threads, from random initialization. Each
  completed 60 optimizer updates. Best validation-loss checkpoints: DR epoch 1,
  quality epoch 2. See docs/MODEL_STATUS.md for exact commands and fitting limits.
- Executed scripts/calibrate.py on 20 dedicated calibration images, including
  separate flip-TTA temperatures. Calibration artifacts are checksum-bound.
  DR fitting NLL/Brier improved while ECE worsened; quality temperature reached
  20. These fitting metrics are not independent test results.
- Executed scripts/evaluate.py --manifest data/processed/deepdrid/pilot_manifest.csv
  --split holdout --output reports/pilot after freezing model and policy choices.
  Holdout: 60 images / 15 patients. DR accuracy 16.67%, macro-F1 0.05714, kappa 0;
  every prediction was grade 3. Quality accuracy 50%; every prediction was poor.
  Useful screening performance and policy superiority were not established.
- Seven policy comparisons across nine budget scenarios produced 51 infeasible
  cases. A joint allocation at review 0.6 / recapture 0.4 gave zero auto coverage
  and sent 7 severe cases to recapture, versus 0 for quality-only/naive policies.
  These unfavorable findings are preserved. No holdout-based retuning occurred.
- Executed scripts/benchmark_cpu.py on 20 validation images, three repeats.
  Source artifacts remain local; sanitized comparisons are in reports/pilot.
  Other development and container activity shared the machine during measurement;
  these timings do not establish isolated performance or strategy superiority.
  Inference forwards one image at a time; RSS is sampled at batch boundaries.
- scripts/run_ablations.py recomputed aggregate comparisons from frozen predictions
  without extra model inference. Its 11 reporting tests passed. The three
  conditional strategies selected identical cases and produced equal outputs.
- Full pytest run with QSART_REAL_MANIFEST set: 67 passed in 37.07 seconds,
  including a genuine trained-model JPEG upload, quality/calibration, TTA,
  batch routing and JSON/CSV export. One upstream AnyIO deprecation warning.
  Later added reporting tests and the final aggregate run are recorded below.
  Ruff checks passed.
- Actual HTTP readiness and prediction on localhost:8001 returned 200 using a
  real DeepDRiD image and second_pass=true. All five Streamlit pages rendered;
  the real response table and export widgets were verified with AppTest.
  The browser displayed the running dashboard. Fixed stale prediction/image
  association and accurately described temporary upload handling.
- Docker Desktop 29.7.2 started; an initial CPU image built. Latest-source
  rebuilds encountered registry DNS and a wheel read timeout. Pinned the observed
  base digest, cached separate dependency layers, and extended pip retries and
  timeouts. Final container execution is pending at this log entry.
- Literature: 43 records and 43 BibTeX entries, retaining all 28 original IDs,
  with source-specific verification limits. Updated the Review-1 brief,
  manuscript draft and Mermaid architecture with actual negative pilot findings.
- Checkpoint code_version records pre-existing HEAD while new source was
  uncommitted. It is not an exact training source identifier; this is documented.
  The held-out report additionally hashes source. Initial implementation commit:
  330bb87 (recorded after pilot execution).
- Raw images, patient manifests, weights, per-image artifacts and Office source
  extracts stay local and are ignored by Git. Source and aggregate reports are
  committed on qsart/research-prototype; main remains unchanged.

## 2026-10-18 — final verification and delivery

- Final aggregate suite with real-image opt-in: 78 passed, no failures/skips,
  127.03 seconds. One upstream AnyIO deprecation warning. Ruff passed.
  Summary: reports/test_summary.json; full JUnit stays local.
- Latest CPU Docker image built after network retry fixes. Both containers are
  healthy. Original host port 8000 belongs to another Python application, so
  Compose ports are configurable and the ignored local .env selects 18000/18501.
  No unrelated application was stopped.
- scripts/verify_deployment.py exercised the actual containers: health, readiness,
  model info, docs, real JPEG prediction, 2-image batch, conditional second pass,
  upload rejection and dashboard health all passed. After restarting the API,
  the same verification passed again, confirming checkpoint persistence.
- Container and host hashes match all 27 runtime Python files. Captured the
  actual container package versions in reports/container_requirements.txt.
- Browser displayed the running container dashboard; saved reports/dashboard_preview.png.
  Removed only the temporary development API/dashboard processes on 8001/8502.
  The verified Docker services remain running on 18000/18501.
- PowerShell launcher syntax parsed successfully. It reads local port settings
  and cleans up the child dashboard when its foreground API exits. The launcher
  itself was not used for final Docker verification.
- Source and aggregate evidence published to qsart/research-prototype; draft PR:
  https://github.com/Riteshhai/Retina-Opthalmology/pull/1 . Main remains unchanged.
- Acceptance remains PARTIALLY COMPLETE for the overall research project.
  The local engineering pilot is verified; useful model performance, complete
  confirmatory experiments, natural-defect modeling and external validation
  remain open. No clinical or scientific superiority is claimed.


## 2026-10-20 — larger development and locked internal confirmation

- Continued in the same local QSART directory. Preserved original assets and
  all v0 manifests, checkpoints and reports. Recovered interrupted transfer/data
  implementation; fixed the official source-split value and finite-config checks.
- Declared docs/research/DEVELOPMENT_V1_PROTOCOL.md before new confirmation
  predictions. Used all 296 unambiguous official-training patients (1,184 images):
  fitting 848/212 patients, selection 168/42, calibration 168/42. Four ambiguous
  subjects excluded. Conservative connected pHash groups stay together.
- Acquired the remaining authorized pinned images with verified metadata, byte
  budget, per-file SHA/readability checks and reuse of local files. Development
  download added about 384.7 MiB; new confirmation added 177.4 MiB. No images,
  identifiers, patient manifests, feature caches or weights are published.
- Reserved 340 official-validation images / 85 patients, excluding all 15 inspected
  pilot holdout patients. No exact/pHash cross-cohort candidates at the declared
  distance. Development SHA 45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d;
  confirmation SHA 21cfd7aa43660a236903430207b27cc3a8366a9129cbcbfa513eee0666baa9b7.
- Downloaded official torchvision ImageNet MobileNetV3-small and ResNet18 weights
  locally, with URLs/full hashes recorded. Code and pretrained-weight licensing
  are distinguished; no broad commercial/redistribution permission is asserted.
- Real frozen-backbone CPU training at 224px used original/flipped fitting
  features, fixed BatchNorm statistics, new linear heads, class-balanced AdamW,
  seed 2026, batch 64, maximum 100 epochs and patience 15. Caches bind actual
  input bytes, manifests, transformations, source and pretrained-weight hashes.
- MobileNet DR: 30 total epochs / 810 updates; best epoch 15 / 405 updates,
  selection accuracy 63.10%, macro-F1 0.53724. ResNet18: 35 epochs / 945 updates;
  best epoch 20, selection accuracy 61.31%, macro-F1 0.51601. The declared
  selection macro-F1/NLL rule chose MobileNet without confirmation predictions.
- Quality MobileNet: 18 epochs / 486 updates; selected epoch 3 / 81 updates,
  selection accuracy 74.40%, macro-F1 0.73726. Actual overall-quality, clarity,
  field-definition and artifact labels were learned. Fixed training-only
  statistical logistic comparator achieved selection accuracy 72.62%.
- Committed transfer source before these runs (51eae27); checkpoints also record
  exact training-source, pretrained weights, feature and input-content hashes.
  `finalize_development.py` audited cross-cohort independence, selected models,
  calibrated on the dedicated 168 images and froze artifact/protocol/evaluation
  source hashes before confirmation. Calibration-fit diagnostics are separate
  from independent prediction performance.
- `evaluate_confirmation.py` validated every frozen hash and inferred 340 images.
  DR: 204/340 = 60.00% accuracy, macro-F1 0.53176, QWK 0.72650, referable AUROC
  0.92482 / AUPRC 0.90789. Patient-cluster accuracy interval: 52.65–67.06%,
  1,000 resamples, seed 2026. Raw/calibrated ECE: 0.08790/0.03999.
- Quality: 74.12% accuracy, macro-F1 0.74028; classical comparator 73.24%,
  macro-F1 0.73133. Defect accuracy: clarity 49.41%, field 48.53%, artifact 35.59%.
  Artifact accuracy is below majority-class accuracy (47.94%); field class 0
  is absent and grade 4 has only 14 images. No defect reliability claim.
- Seven methods / nine fixed budget pairs: 27 of 63 infeasible. At actual review
  .4 / recapture .2, quality-only/naive/joint missed-referable counts are 12/5/5,
  severe recapture 18/18/0, automated coverage .4 each. Exploratory paired patient
  intervals are conditional on fixed assignments, without multiplicity correction.
  No model, temperature, proxy-cost or threshold tuning followed confirmation.
- Supplementary post-confirmation analysis reuses the locked predictions, checks
  cohort/lock identity and reproduces the confusion matrix before 97 natural-code
  strata. Poor/good accuracy 58.51%/61.84%, referable false-negative rates
  18.92%/20.00%, entropy error AUROC 0.76521. This does not establish universally
  higher false negatives with poor quality. Fixed a pandas 3 read-only mask error
  in the new aggregation script and reran it successfully; no inference changed.
- Ran five CPU strategies on a deterministic 48-image selection prefix, three
  repeats, two threads, i5-12500H, model-forward batch 1 / cohort chunk 8. Mean
  ms/image: single 194.72, all TTA 188.15, uncertainty 182.78, severity 169.51,
  budget 182.29. Training and confirmation inference had finished; normal
  workstation/report activity remained. Sequential strategy order, excluded
  decode/HTTP/triage and batch-boundary RSS (~1,015–1,016 MiB) prevent a reliable
  speed claim. The selection subset is not confirmation. Published aggregate
  strategy performance/workloads and all timing qualifications.
- Full suite with QSART_REAL_MANIFEST=confirmation_v1.csv and all three selected
  checkpoint/calibration environment variables: 89 passed, 39.07 seconds, one
  upstream AnyIO deprecation warning. Ruff and Compose configuration checks pass.
- Rebuilt `qsart:local`, configured ignored .env to selected v1 artifact paths,
  kept old pilot weights, and verified actual Docker HTTP JPEG single/batch/TTA,
  readiness/info/docs, unsupported upload and dashboard health. Card status is
  matched; wrong weights or temperatures suppress recorded metrics. All 31
  container runtime Python hashes match host; actual dependencies captured.
- Browser showed current internal results/model version; saved
  reports/dashboard_v1_preview.png. Dashboard and API remain on 18501/18000.
- Restarted the API container and repeated all real HTTP checks successfully;
  reports/deployment_v1_after_restart.json confirms selected artifact persistence.
- Updated README, current Review-1 brief, manuscript, architecture and acceptance
  evidence. Historical pilot status documents are marked as historical. Public
  reports contain aggregate evidence only. The new and old protocols/cohorts
  differ; no paired accuracy-improvement estimate is claimed.
- Overall research acceptance remains PARTIALLY COMPLETE. External data/transport,
  independently developed comparator tuning, repeated seeds, controlled quality
  stress tests, stronger defect models, patient-level referral and prospective
  specialist/recapture outcomes remain open. Current confirmation is consumed
  evidence and must not become an undisclosed tuning target.


## 2026-10-20–21 — MASTER_SPEC_v1.0 adoption and execution

- Adopted the exact new user specification in MASTER_SPEC.md and a versioned
  copy; SHA256 1b6af695af482d43d763e7e5cb74d4fe1002f3778fba8abc55872500a5a8c275.
  Created CHANGELOG.md, specification alignment and current status; updated
  AGENTS/TASKS/README. Preserved original assets and all v1 measurements.
- Preserved exact pre-adoption Git source cedf182 in an ignored local archive;
  recorded archive SHA in docs/SPEC_ALIGNMENT.md. Historical replay uses that
  source rather than rewriting the old lock after source extension.
- User specification section26 explicitly authorized parallel workstreams.
  Separate ownership covered triage, conformal/base-paper study, data/models and
  root API/deployment integration. No agents overwrote another's owned files.
- CR-001 implemented current AUTO_NONREFERABLE semantics in new triage_v2.py;
  old triage.py unchanged. Predicted referable/severe risk, poor quality,
  uncertainty and conformal guards restrict clearance. Exact independent floor
  capacities and structured BudgetOverloaded diagnostics never force clearance.
  True referable clearances are misses; deferred care remains unresolved.
- Read the full selected primary paper and authors' code; recorded the critical
  study and updated P24 in the 43-record literature matrix. Independently
  implemented LAC and deterministic adjacent ordinal construction, exact kth
  quantiles, ties, patient-max scores and bound JSON. Public-code quantile/batch
  discrepancies and GPL author-code exclusion are recorded. No exact randomized
  replication or external coverage guarantee is claimed.
- Ran scripts/calibrate_conformal.py with separate 84-image/21-patient temperature
  and 84/21 conformal groups. Independent DR temperatures: 1.251752 first-pass,
  1.262026 all-flip. The final sidecar uses the same raw-softmax transformation
  during fitting and inference; an initial pre-integration artifact was retained
  locally, with the corrected reproducible artifact named conformal_r1.json.
  Patient/photo/pHash overlap audit passed and all 168 actual image hashes matched.
  Existing evaluated grade/quality probabilities and all v1 artifact bytes remain.
- Ran scripts/prepare_research_scores.py on 168 real selection images/42 patients,
  for first-pass, 25%-budget and all-flip profiles. Private score cache remains
  under checkpoints/spec_v1. Public development diagnostics: LAC image coverage
  94.64%, mean set size 3.345 and all-views patient coverage 85.71%. Ordinal quantile
  1.0 gives every grade for every image, 100% coverage but no information.
  Independent audit reproduced all 168 sets in every profile and found no
  credible algorithm/integration bug; the unfavorable outcome is retained.
- Evaluated all 360 declared profile/budget/policy combinations:288 overloaded;
  all 72 feasible combinations review every image and have zero auto coverage.
  No thresholds/costs were retuned to hide this failure. These are development
  diagnostics, not a fresh test, triage benefit or clinical safety result.
- Integrated typed /triage/config, /triage/simulate, /research/experiments and
  /research/metrics, with current prediction/schema policy version 2.0. Simulation
  only admits cached selection datasets; supplied score experiments are labeled
  caller-supplied/exploratory. Configured mismatched/missing CP fails readiness.
- Eight dashboard modules now include actual CP/reasons, selection-only simulator,
  real-model comparison and experiment explorer, session quality/latency metrics,
  quality/risk filtering and exports. Browser displayed all 8 modules and actual
  overload. No patient-image screenshots were committed.
- Added and executed actual EfficientNet-B0 frozen-feature training on 848 fitting
  images, with168 selection images, seed 2026. 28 epochs/756 updates, selectedepoch 13:
  accuracy 64.29%, macro-F1 0.524924, NLL 1.271143. MobileNet retains .537238 macro-F1
  under the declared rule. No APTOS/confirmation evaluation or model switch.
  Official pretrained SHA verified; exact training source/input hashes recorded.
- scripts/build_model_comparison.py recomputed all three genuine candidates'
  selection metrics from saved probabilities/labels, plus actual model sizes.
  Unmeasured candidate latencies remain null; legacy MobileNet latency is scoped
  separately. Five private-copy corruption/provenance checks passed.
- Strict local APTOS adapter/inventory implemented: source labels, SHA/pHash
  groups, conflicting-label rejection and no invented patient/quality fields.
  Added split_group_id leakage guard to the common reader. The user confirmed
  APTOS is not downloaded; no credentials/access bypass or mirror substitution.
  APTOS training remains BLOCKED; acquisition steps are in docs/APTOS_SETUP.md.
- Initial full integration run exposed 2 obsolete tests expecting string overload
  errors. Updated them to assert current structured code/count/shortfall semantics.
  Final full real-model+CP suite: 198 passed, no skips/failures, 78.51s, one upstream
  AnyIO warning. Ruff/Compose passed. Subsequent default chart metric check:
  1 passed,4 deselected,8.87s. Launcher syntax passed; selected container artifact
  paths now map to local Python paths, but the launcher was not used as proof.
- Rebuilt actual Docker CPU services and verified new real JPEG single/batch/TTA,
  CP/card matching, readiness/docs, unsupported upload, real cached simulation
  evaluated/overloaded and research routes. Local URLs remain 18000/18501.
- scripts/benchmark_deployment.py measured 12 selection JPEGs, two warmups excluded,
  full loopback HTTP preprocessing/model/CP/routing/JSON chain: mean first-pass
  request 304.64ms, extra-pass 210.44ms; batch 4 budget 25% 1004.29ms/request
  (251.07ms/image). API kernel peak RSS 644,980 kB since startup. Fixed order,
  small sample and live-machine variability prevent a speed-superiority claim.
  Training and full tests had finished before this timing run.
- Formal manuscript work remains deferred under the adopted specification.
  Existing draft/Review1 evidence is preserved and clearly historical. Overall
  research remains IN PROGRESS; useful conformal/triage, independent comparator
  tuning, repeated seeds, controlled degradation and external IDRiD still open.

- Final Docker source verification matched all 37 runtime Python files; captured
  actual container dependencies. Repeated real HTTP/conformal/simulator checks
  after API restart passed. Specification-byte, public-aggregate privacy and local
  documentation-link audits passed. Exact new report/spec bytes are preserved by
  scoped Git attributes, so their recorded hashes survive checkout.
