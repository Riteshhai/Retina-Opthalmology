# Execution log

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
