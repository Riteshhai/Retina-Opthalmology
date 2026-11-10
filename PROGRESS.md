# Execution log

## 2026-11-10 — completed the two-seed APTOS resolution comparison

- The first seed-2027 384-pixel worker exited with I/O errors and eight
  intact checkpoint and score pairs. Archived its state and logs, restored
  Docker and reran the locked input preview. Ran
  `./scripts/run_aptos_384_docker.ps1 -Seed 2027 -ImageSize 384 -Execute -Resume`
  at source `0a8eb4a5f29156ebf0032b517bd809e51060c020`. The trainer verified
  the lock and checkpoint chain, then resumed epoch 9. It exited successfully
  under the patience rule and selected epoch 5. The 224-pixel arm selected
  epoch 6 after 10 epochs. Released the temporary Windows execution request.
- On the 521 development selection images, seed-2027 selected accuracy was
  81.38% at 224 pixels and 82.53% at 384 pixels. Macro-F1 was 0.693 and 0.681,
  and grade 3 recall was 15/26 and 12/26. Both seeds favored 384 pixels in
  accuracy and macro-F1 at equal updates. All eight paired intervals include
  zero. The reserved test and external populations supplied no repeat results.
- Ran `python scripts/summarize_aptos_resolution_repeat.py`, the pinned
  Linux selected-checkpoint replay, and the independent host audit. The audit
  verified 19 checkpoints, 9,899 saved selection rows, nine matching RNG
  epochs, 2,636 source image hashes, 352 preserved artifacts and all eight
  intervals. Both selected replays matched the saved logits exactly.
- Ran `python -m pytest -q tests/test_aptos_resolution_match.py
  tests/test_research_api.py tests/test_dashboard.py` and
  `python -m ruff check .`. All 39 targeted tests passed in 92.37 seconds
  with one upstream AnyIO warning. Ruff and Compose configuration passed.
  The locked source also retains its earlier 634-test local run and GitHub CI.
- Rebuilt the QSART Docker services. Both were healthy and all 73 runtime
  files matched. The deployed model passed genuine-image single, batch,
  conformal, simulator and overload checks. The live API and Streamlit
  explorer matched all eight checkpoint rows. Independent EyeQ quality
  validation still requires the official photographs.
  [Results and verification](docs/research/APTOS_RESOLUTION_SEED2027_V1_RESULTS.md).

## 2026-11-10 — locked the second APTOS resolution pair

- Fixed seed 2027 before fitting. The 224- and 384-pixel arms reuse the audited
  source cohort, pinned ImageNet weights, exact trainer and original Docker
  image. Each worker verifies its original image bytes and reads the preserved
  resize cache. The new outputs are separate by resolution.
- Declared the two selected-checkpoint and epoch 5 comparisons in the same
  384-minus-224 direction and with the same fingerprint-component draws. This
  repeat scores the fitting and selection roles for grading. The seed-2026
  temperature and conformal evidence remains preserved.
- Both read-only previews passed with `./scripts/run_aptos_384_docker.ps1 -Seed 2027`
  at `-ImageSize 224` and `-ImageSize 384`. Ran `python -m pytest -q
  --junitxml=reports/aptos_resolution_seed2027_v1_full_tests.xml` with the genuine
  DeepDRiD validation pipeline. All 634 tests passed in 157.92 seconds with one
  upstream AnyIO warning. Ruff, launcher syntax and Compose configuration passed.
  [Locked protocol](docs/research/APTOS_RESOLUTION_SEED2027_V1_PROTOCOL.md).

## 2026-11-09 — completed and audited the matched APTOS resolution comparison

- Ran `./scripts/run_aptos_384_docker.ps1 -ImageSize 224` and then added
  `-Execute` on clean source `cc4f4450cb9b2663def6cb1e637949cdfdd50782`.
  The 224-pixel worker completed 15 epochs and 3,975 updates, selected epoch 12,
  and exited successfully. Training took 2,828.64 seconds with 1.07 GB sampled
  process RSS. The same worker continued after Windows standby and the temporary
  system execution request was released at exit.
- On the 521 development selection images, the selected 224/384 checkpoints had
  83.11%/83.49% accuracy, 0.715/0.697 macro-F1 and 15/26 versus 8/26 grade 3 recall.
  At epoch 5, both arms had 1,325 updates and 78.31%/80.42% accuracy. All four
  paired intervals include zero. The cohort supplied checkpoint selection and
  the runs use one seed.
- The audit verified 15 immutable epochs, 7,815 saved selection rows, 13 common
  epochs' matching RNG states, all 14,588 raw and fallback sets, all four paired
  intervals and 350 preserved artifacts. It reconstructed the initial head in
  the pinned Linux image and matched the selected checkpoint replay exactly.
- Ran `python -m pytest -q --junitxml=reports/aptos_224_matched_v1_full_tests.xml`
  with genuine DeepDRiD validation artifacts and the conformal sidecar. All 631
  tests passed in 120.20 seconds with one upstream AnyIO warning. Ruff and Compose
  configuration passed. Source CI passed with 630 tests and the real-image test
  skipped on GitHub.
- Rebuilt and exercised the primary Docker services with the deployed model card
  and conformal sidecar. All 73 runtime files matched. Actual single, batch and
  simulator checks passed, and Streamlit AppTest matched the four checkpoint
  rows and four conformal views to the live API.
  [Results and verification](docs/research/APTOS_MATCHED_RESOLUTION_V1_RESULTS.md).

## 2026-11-09 — locked the matched 224-pixel arm

- Pinned the 384-pixel evaluation, selected checkpoint and epoch 5 scores.
  The new arm binds the same data, initialization, training recipe and Docker
  image. Verified 51 shared scientific dependency hashes and the initializer
  syntax tree. The trainer change derives the checkpoint name from input size.
- Declared selected-checkpoint and fixed epoch 5 contrasts, both in the
  direction 384 minus 224, with 2,000 paired fingerprint-component resamples.
  Epoch 5 compares 1,325 updates per arm. The development data already supplied
  checkpoint selection, and the run uses one seed.
- Ran `python -m pytest -q tests/test_aptos_384.py tests/test_aptos_resolution_match.py tests/test_research_api.py tests/test_dashboard.py`.
  All 45 tests passed in 60.91 seconds with one upstream AnyIO deprecation
  warning. Ruff and the Compose configuration check passed. Fitting and the
  measured comparison follow this locked source.
  [Matched protocol](docs/research/APTOS_MATCHED_RESOLUTION_V1_PROTOCOL.md).

## 2026-11-09 — completed and audited the 384-pixel APTOS run

- Executed clean source `a446d647535d629f03f46ea2b43a73e29523fff7` in the
  nonroot Docker worker with four CPUs, 4 GiB of memory and read-only inputs.
  The patience rule stopped at epoch 13 after 3,445 optimizer updates. Epoch 9
  reached 83.49% selection accuracy, 0.697 macro-F1 and 0.907 quadratic kappa.
  Training took 7,400.51 seconds and peak sampled process RSS was 1.72 GB.
- The prior 224-pixel run had 80.81% accuracy and 0.652 macro-F1. Both paired
  intervals include zero. Grade 3 recall was 8/26 and referable AUROC was
  0.9747. Temperature 1.560740 lowered selection NLL from 0.5645 to 0.4831.
  At alpha 0.10, raw LAC inclusion was 91.36% overall and 14/26 for grade 3.
  The evaluation retains the tradeoffs across all 14 conformal settings.
- The separate audit reconstructed epoch metrics from saved scores, the ImageNet initialization,
  14,588 individual raw and fallback sets and four paired intervals. All
  340 prior artifact hashes match. The selected checkpoint replay reproduced
  the saved logits exactly. The comparison bundles several training changes.
- Ran `python -m pytest -q --junitxml=reports/aptos_384_v1_full_tests.xml` with
  the genuine DeepDRiD validation artifacts and conformal sidecar. All 627 tests
  passed in 206.54 seconds with one upstream AnyIO deprecation warning. Source
  CI passed with 626 tests and the local real-image test skipped in CI.
- Rebuilt the main Docker app and checked 72 source and configuration files
  against the checkout. The live service passed real-image prediction, batch,
  conformal, simulator and overload checks. The explorer exposes the new
  report and verified both alpha settings and both population views.
  See [measured results](docs/research/APTOS_384_V1_RESULTS.md).

## 2026-11-09 — locked the 384-pixel APTOS development protocol

- Verified the frozen 2,115 fitting, 521 selection and 527 calibration images,
  the saved comparator scores and the full hash of the cached ImageNet weights.
  The input check passed on Windows and in the Docker worker.
- Fixed seed 2026, the direct RGB resize and augmentation recipe, weighted
  AdamW, the 15-epoch cosine schedule and accuracy-based checkpoint selection.
  Added disk streaming and immutable epoch recovery with separate score hashes.
  The worker uses four-image microbatches, eight-image optimizer windows, four
  CPUs, a 4 GiB container limit and read-only source mounts.
- The trainer, inference, research API and dashboard checks passed with 54
  tests in 74.55 seconds. After the cache and checkpoint publication changes,
  the final trainer and real-image integration gate passed 8 tests in 12.15
  seconds. Each run reported one upstream AnyIO deprecation warning. Ruff and
  the Compose configuration check passed.
- Added the 384-pixel research explorer and restricted its direct-resize
  checkpoint to the research scorer. The protocol and execution files are
  committed together before fitting. Study results and the audit follow the run.

## 2026-11-06 — completed cache acquisition and restored primary Docker app

- The resumed four-worker CLI completed all 14 pinned Parquet files, 6,535,643,877
  bytes and 35,108 unique names. Every file SHA256/size, cached card and local receipt
  verifies. Original failed long-response logs and source-bound recovery records
  remain private. The automatic metadata join decodes no images and fits no model.
- Exact EyeQ join: 12,529 of 12,543 author-train rows matched; 14 missing, all Reject.
  None of 16,249 author-test images occur. Matched qualities 8347/1876/2306 and
  missing 8471/4558/3234 (Good/Usable/Reject), zero shared DR-label conflicts.
  22,579 mirror rows have no EyeQ annotation. Original source partitions remain
  external-only; no transported quality-truth or independent-validation claim.
- Independent auditor rehashes every shard, all 307 prior artifacts and source
  CSVs; reconstructs all 35,108 cached rows and 28,792 annotation/private-join rows;
  and matches all identities/DR labels to the earlier 1,291,755-byte remote scan.
  Corrected only private audit setup for the CSV's unnamed index and an HTTPS
  false positive in the local-path guard; preserve stopped attempt logs. Footer
  read-ahead may include adjacent compressed payload, with no JPEG decoding.
- Separate deterministic three-payload format/geometry checks and all 14-shard
  no-network cache reuse are recorded separately; no quality/model reassessment.
  Patient photos/IDs/Parquet/checkpoints remain local and ignored.
- Docker engine 29.9.2 was started and the PyArrow 25.0.1 image genuinely rebuilt.
  All 70 source/configuration files match. Temporary 28000/28501 checks passed;
  verified owned native service commands, stopped those and started primary
  qsart Compose on 18000/18501. Actual JPEG/batch/TTA/conformal/simulator/overload
  checks pass, both services healthy/nonroot/read-only. Removed only the temporary
  project after primary verification; unrelated user containers remain untouched.
- Current 618 full/25 targeted test results are retained from the tested C1 source;
  no subsequent scientific/API/UI source change. Added public aggregate cache,
  runtime and CPU feasibility evidence plus updated setup/backlog/status notes.
  Exact source C1 CI 42259048809/run 56 passed. Final results publication and its
  exact-head CI are recorded privately after push. Full EyeQ test/quality truth,
  useful triage and full 384px training remain open; manuscript remains deferred.

## 2026-11-06 — completed bounded 224/384 CPU feasibility

- Executed from clean c1e52fd21f43e38de78007e1376055276be5865d on 32 hash-selected
  fitting representatives, with original source image/checkpoint hashes verified.
  Both fresh workers completed four actual AdamW updates, batch8/threads2/FP32,
  direct RGB resize and active BatchNorm. All losses/gradients remained finite;
  temporary parameters changed. No updated weights or predictions were saved.
- Four update times:224px2.71586s/384px8.24752s; sampled worker/launcher process-tree
  RSS1.30977GB/2.74216GB. Pixel preparation2.98046s/3.00417s excludes model loading.
  These concurrent-workload observations establish short-run feasibility only,
  not controlled speed, full-training memory/duration, convergence or accuracy.
- Saved-record audit verifies all 32 selected fitting inputs/byte hashes, class
  weights, both raw worker records, source/protocol Git bytes, original checkpoint
  and 307 historical artifacts. No independent timing/gradient repeat is claimed.
  Report SHA d43e9d5e00ed870523a90b3ee525fc6cad4a9f20d5285fd179a37f89117a1620.
- Final full suite after transport/cache-root/probe changes:618 passed130.82s,
  zero failures/errors/skips, genuine production/conformal integration and one
  upstream AnyIO warning. Final25 targeted passed11.29s. Earlier613/615/617 runs
  remain historical phase evidence. No post-test scientific/API source changes.
- Added feasibility/audit reports and results notes. Full384px paper training and
  independent quality validation remain open. The user-selected mirror transfer
  continues with source-bound partials; no mirror image has been decoded/predicted.

## 2026-11-06 — transfer recovery and bounded 384px probe preparation

- Long mirror responses repeatedly closed early. Two Parquet shards passed exact
  SHA256; two partials are preserved with source-bound markers. The initial transfer
  stopped after its retry budget, with no corrupt file promoted or model use.
- Add 32-MiB range requests, twelve consecutive network retries and optional
  one/two/four file workers. Regression fixtures cover multi-chunk completion and
  a disconnect followed by resuming the actual written prefix. The source pin,
  expected bytes and final SHA256 remain unchanged; old partials remain reusable.
- Found the new cache directory was not covered by legacy ignore rules; no payload
  was staged/uploaded. Explicitly ignore data/external and require that root in the
  CLI, with a regression check. Keep all payloads/card/IDs/receipts local.
- Add a predeclared 224/384 CPU feasibility probe using 32 hash-selected fitting
  representatives, a temporary copy of the genuine source model and four weighted
  AdamW steps per fresh worker. It exports no weights/predictions and uses no
  selection/calibration/test/external pixels. Full-paper training remains separate.
- Probe fixture tests passed2/16.66s; corrected its import ordering after Ruff found
  two E402 errors. Earlier expanded suite615 passed107.85s before range recovery;
  range-recovery suite617 passed106.08s, zero failures/errors/skips, one upstream
  warning. Final25 targeted checks passed11.29s after the additional ignored-root
  regression. Memory sampling explicitly includes venv descendants and verifies
  the actual probe worker. Ruff/whitespace pass before source commit and resumption.

## 2026-11-06 — user-selected Hugging Face acquisition preparation

- The user deferred manual EyeQ acquisition and selected bumbledeep/eyepacs for
  an online execution path. Pinned mirror revision 3e94599d7e41d9992a8362cc0467eb397e11210f,
  14 Parquet files/6,535,643,877 bytes/35,108 declared rows, file sizes/LFS SHA256 and
  card hash. Card describes transformed images and removed nearly black scans;
  its MIT claim is recorded separately from unverified underlying image terms.
- A 66,599-byte metadata-only range probe found original JPEG names in image.path.
  No embedded JPEG decoding, full-cohort join or model inference was performed.
  Initial broad discovery found an empty EyeQ-named repository; the user's exact
  link resolves that ambiguity. No alternative source is substituted.
- Added a plan-first explicit download CLI, resumable identity-bound partials,
  SHA256 cache reuse, trusted HTTPS redirects, strict range/size/path guards,
  read-only Parquet metadata projection and separate original EyeQ quality/DR join.
  App startup never downloads data. Source/truth/pixel independence readiness stays
  false; cached payloads/IDs remain ignored. Pin the existing PyArrow 25.0.1 dependency.
- All 20 targeted generated-fixture tests passed in 8.82s; Ruff and default source/
  space plan pass. Full genuine-model/conformal suite passed 613 tests, zero
  failures/errors/skips,117.96s (JUnit117.941s), one upstream AnyIO warning.
  Commit the tested source/manifest/plan before actual cache acquisition.
- The previously completed APS/RAPS publication 49fc4384a0bfbd43a63c1eaf69e61109075c853a
  passed exact-head CI42088242301. Its scientific results remain unchanged. The
  later checks found native services/Docker unavailable; the native launcher was restarted and
  current Linux-engine pipe availability will be reported separately.

## 2026-11-06 — completed fixed APS/RAPS score comparison

- Executed all 34 fixed saved-logit comparisons from clean
  e7fd2f1b85d79e2f9f910433eea0dd7beea06b7c. New score thresholds use 254 original
  conformal representatives; source classifier/temperature and 521 selection rows/
  507 representatives stay fixed. No new images, neural inference, classifier/
  temperature fitting, RAPS tuning, external/test inputs or production update.
- Primary alpha .10/uniform seed 2026/raw RAPS-minus-APS mean size -.09980806,
  descriptive 95% interval[-.13205517,-.06896552]. Overall inclusion delta -.00575816
  includes zero, while grade-3 inclusion falls 18/26 to 14/26: delta -.15384615,
  interval[-.30434783,-.03448276]. No superiority or clinical claim. Randomized
  ordinal increases grade-3 inclusion to 25/26 but lowers grade-4 inclusion to 30/40
  versus APS 32/40, with wider sets. All unfavorable/fallback/seed results remain.
- Independent loops/counts oracle reconstructed 34 thresholds (maximum gap
  1.110223e-16), all 136 metric views, 35,428 raw/fallback set rows and 160 paired
  intervals. All 14 historical sets/uniforms and 295 preserved artifacts match.
  A separate check reconstructs 20 conditional summaries/120 mean-SD cells.
  Aggregate SHA a883d166fe7172d77ee615b300ae46e802355c19731ecf9080640c3b41fc0e88.
- All 593 tests passed before execution in 109.14s, zero failures/errors/skips,
  genuine production/conformal integration; 35 targeted in 12.89s. One upstream
  AnyIO deprecation warning. Same scientific/API/UI source retained; Ruff/Compose/
  whitespace pass. Source CI42086591713 passed at the exact execution commit.
- Actual native and rebuilt nonroot/read-only CPU-container JPEG/batch/TTA/
  readiness/conformal/simulator/overload pass. Live-API AppTest checks 20 records,
  all 40 adaptive control combinations, 2,200 metric cells and 1,280 interval cells
  in each runtime. All 68 container source/config files match. Primary figure
  visually inspected. Host AppTest against actual HTTP is separate from a new
  interactive browser-rendering check. Temporary test containers removed; native
  app remains on 18000/18501 with original production card/sidecar matched.
- Preserve Matplotlib's generated SVG path spacing with a scoped Git whitespace
  attribute added after figure verification. No scientific code/result/figure bytes
  change; staged whitespace checks pass after the attribute correction.
- Local implementation/config/protocol/tests/API/UI, nine aggregate/audit/runtime
  JSON reports, PNG/SVG and results notes are saved. Fixed score comparison
  VERIFIED COMPLETE; exact/tuned/384px paper reproduction, useful triage and
  independent quality validation remain IN PROGRESS. Next executable source task:
  lock 384px training/compute feasibility while preserving reserved tests. EyeQ
  still needs original photos/terms/storage/overlap evidence; formal writing deferred.

## 2026-11-06 — fixed APS/RAPS comparison preparation

- Found a local EyeQ author-repository copy: seven metadata/source hashes match
  the already pinned audit, but it contains only five example JPEGs and no full
  EyeQ image cohort. Read only directory metadata and seven non-image source files;
  no example image was decoded or used as independent validation.
- Added independently expressed APS/RAPS ranked-mass scores, stable probability/
  grade ties, exact ranks, explicit uniforms and research-only artifact validation.
  RAPS thresholds above one remain valid. Fixed initial-author CFG k_reg=2/lambda=.1;
  no tuning or exact-paper-reproduction claim. Historical sidecar code stays fixed.
- Locked one saved APTOS source model/temperature, 254 calibration representatives,
  521 selection rows/507 representatives, two alphas, five shared historical uniform
  streams, raw/fallback results and all 160 paired component-bootstrap intervals.
  Fourteen historical LAC/ordinal controls must reproduce. No new images, neural
  inference, classifier/temperature fitting or reserved-test/external input is used.
- Input-only readiness passed and snapshots 295 historical/master/production/source
  artifacts. New score outcomes have not been generated. Added API/explorer controls
  and protocol/config before actual execution. Targeted tests:35 passed,12.89s,
  one upstream AnyIO deprecation warning; Ruff/Compose/whitespace checks pass.
  Full genuine-model/conformal suite:593 passed, zero failures/errors/skips,
  109.14s (JUnit109.126s), same upstream warning. Added a saved-report-only plotting
  script; Ruff/Compose/whitespace pass. Commit tested sources before new outcomes.

## 2026-11-05 — completed EyeQ metadata/access audit and current Docker verification

- Executed the metadata-only audit from clean 9f7f22f79362090039369f8a34625d326189c6e2.
  All 28,792 rows/20,662 filename subject keys, both external-only splits and three
  original quality categories are preserved. No EyeQ image was acquired/decoded/
  predicted; no model, temperature, conformal, policy or production update occurs.
- Standard-library reconstruction passed all 28,792 annotation/manifest/group rows,
  30 quality/DR cells, four binary population views, four table comparisons, seven
  source objects/independent GitHub tree, receipt and two comparative terms files.
  Literal raw index header is blank; pandas calls it Unnamed: 0. The auditor's header/
  Windows UTF-8 handling was corrected before export; source bytes stay unchanged.
- All 271 historical/master/production artifacts match. Public aggregate SHA256
  e0941a2319941e28052c614b68a618d5fe9ed850f71cc7b3c3e30e6b7755b645.
  The stopped earlier attempt's private manifests remain preserved and byte-identical.
  Source CI42080027323 passed at the exact execution commit. All 558 integrated/22
  targeted tests passed before execution; JUnit suite times 102.731s/11.840s,
  zero failures/errors/skips. Same tested scientific/API/UI source remains fixed.
- Actual native JPEG/batch/TTA/conformal/readiness/simulator/overload checks pass.
  Live-API AppTest matches 19 records, both source splits, three quality categories,
  all three nonzero table disagreements and explicit metadata-only/readiness captions.
  Restarting the old API launcher closed its attached dashboard; standard launcher
  restored both on 18000/18501, then actual deployment checks passed.
- Docker initially reported a missing Linux-engine pipe, then became available.
  `docker compose -p qsart-eyeq-v1 up --build -d` with temporary 28000/28501 ports
  rebuilt current source and started healthy nonroot/read-only CPU services.
  Actual container JPEG/batch/TTA/conformal/readiness/simulator/overload and 19-record
  explorer checks pass; all 65 runtime source/configuration files match local bytes.
  Runtime image identity and installed packages are recorded. No new interactive
  browser-rendering check. Historical Docker failure records remain preserved.
  Temporary verification containers were removed after checks; the native app
  remains running on 18000/18501.
- Added eight aggregate/audit/native/container verification reports and results
  notes; updated setup, README, status/backlog and change log. Original images,
  identifiers, author source copies, private manifests, weights and XML remain local.
  Official photographs/accepted terms/storage, cross-source pixels and a locked
  harmonization/evaluation protocol are next. Metadata audit VERIFIED COMPLETE;
  independent quality validation and overall research remain IN PROGRESS.

## 2026-11-05 — EyeQ public-provenance export correction

- The first clean-source execution at 0c6124f6779f9bf21faef5f91934f684bad60c04
  stopped before public report creation: the inherited privacy guard mistook the
  final `s:/` in an HTTPS provenance link for a Windows drive path. No photograph
  was decoded and no model was run. Preserve the two partial private manifests
  and their hashes in an ignored attempt directory; retain the source receipt.
- Use an EyeQ-specific guard that distinguishes public HTTPS source links from
  local paths and rejects credential URLs, unverified hosts and private identity
  keys. Historical study helpers and measured artifacts remain unchanged.
- `python -m pytest tests/test_eyeq.py tests/test_eyeq_explorer.py -q` passed all
  22 tests; JUnit suite time 11.840s. The full suite with genuine production and
  conformal artifacts passed 558 tests, zero failures/errors/skips; JUnit suite
  time 102.731s. XML evidence is retained locally. `python -m ruff check .` and
  `git diff --check` pass. Commit corrected source before restarting the audit.

## 2026-11-05 — EyeQ source/annotation audit preparation

- Rechecked primary author source and paper v4, pinned EyeQ1dcfabb08cd31045de1a4a3ee82e48f6833697aa.
  Downloaded only seven public label/code/README files and separately archived
  pinned DDR README/LICENSE for terms comparison. No example photos, weights,
  author result CSV, source photograph, account credential or terms acceptance used.
- Preparation CSV inspection:28,792 rows/20,662 filename subject keys, zero duplicate
  IDs/shared split keys. Retain train Good/Usable/Reject8347/1876/2320 and test
  8471/4558/3220. Document README train-Usable typo and paper/README test one-image
  disagreement without changing labels. Both eyes stay grouped by source key;
  filename counts do not establish cross-source pixel or patient independence.
- Implemented pinned metadata parser, external-only/training-ineligible manifests,
  separate source quality/DR labels, explicitly unselected binary sensitivity views,
  exact-JPEG private inventory and hash-bound public metadata fetch. Added source/
  terms/space/readiness explorer, fixtures, config and protocol. Full original-image
  listing exceeds current project-volume free space; selected/unpacked footprint
  and direct acceptance terms remain unverified. No gated image download attempted.
- All21 targeted tests passed11.02s; full557 passed102.86s, zero failures/skips,
  genuine production/conformal integration. Ruff,Compose and whitespace checks pass.
  Metadata-only readiness passes seven source hashes; preservation preview271 old
  artifacts. Source/config/protocol committed before frozen report execution.
  Final standard-library reconstruction/native integration remain pending here.

## 2026-11-05 — completed quality calibration and stress v1

- Executed from clean 42ae8621649b11d666f7b5e3e8516b0ac3a5a812. All six fixed
  quality models completed 24 calibration-role head temperatures and 8,064 single-pass
  model-image observations, with only 336 source images decoded. No classifier,
  conformal, policy or production update. Historical calibration-role and repeated
  selection reuse are disclosed; no confirmation/IDRiD/APTOS or fitting pixels used.
- Primary adapted mean clean-selection NLL 1.02045741 to .87411869; delta-.14633872,
  descriptive95% interval[-.21830587,-.07930656]. All three adapted point NLL changes
  are negative; all three frozen changes positive. Conditional adapted mean NLL
  .99734034 to .87326385, frozen .95549177 to .96154892. Keep every adverse fit.
- Positive temperature preserves all class confusion matrices/accuracy/F1/recall.
  Primary field ECE .12950 to .17579 and Brier .43208 to .45714 worsen despite lower
  NLL. Rare-field0/1 and adapted artifact0/6 failures persist. Scaling does not
  repair discrimination, provide new quality supervision or establish triage utility.
- Primary scaled poor-score changes blur2 +.08822287, interval[.04974196,.12779639],
  JPEG20 +.04371024,[.00771073,.07743241]. No new corrupted-quality truth exists;
  these are responses only. Primary scaled adapted-minus-frozen blur response
  -.12577232,[-.15538536,-.09755179]; JPEG20 contrast+.01490227 interval includes zero.
- Independent audit passed24 scalar objectives (maximum NLL gap to independent
  bounded optimizer3.996714e-9),96 clean head metric views,486 declared intervals
  and 48 zero controls. All 8,064 raw logits replay exactly; all model state stays
  fixed. All 336 source hashes/eight pixel caches,26 independent geometry images/
  104 transform cells and 234 historical/master/production artifacts match. Scientific
  report SHA906322ccd6a28a5a330a935573af6ac775de5e191eafa474ebd28f96c04b6bc3.
- All 536 local tests passed in 100.90s before fitting/inference, zero failures/skips,
  genuine production/conformal integration;14 targeted passed in 14.06s. Same scientific/
  API/UI source retained. Ruff/Compose/whitespace pass. Source CI42071805872 passed.
- Actual native JPEG/batch/TTA/readiness/simulator/overload pass after restart.
  Live-API AppTest verifies18 records,48 clean model/head/role views and 48 stress
  model/state/response views, all values/supports, fixed primary and fitting/truth
  captions. Both figures visually inspected. Docker Linux engine unavailable;
  no new browser/container proof. Overall project IN PROGRESS; formal writing deferred.
- Local new module/runner/config/protocol/tests/API/UI, plotting script, aggregate/
  audit/native/verification/figures and results notes are retained. Next highest
  executable gate: independent quality-data source/terms/label access audit before
  proposing broader validation or production changes. Current models remain fixed.

## 2026-11-05 — quality calibration and stress v1 preparation

- Declared all six fixed quality checkpoints,24 scalar temperatures fitted only
  on the separate 168-image/42-patient calibration role, clean selection diagnostics
  on 168/42 and seven exact canonical-input conditions. Disclose historical role
  reuse; no classifier, conformal, policy or production updates. No fitting pixels,
  consumed confirmation/IDRiD or APTOS images enter the new study.
- Altered quality images have no new annotations: report probability/ordinal-code
  response changes only, never corrupted quality accuracy. Predeclare fixed-primary
  mean NLL change, blur2/JPEG20 poor-score response, all 486 paired intervals and 48
  zero controls, without refitting/reselection/multiplicity adjustment.
- Added strict role/pixel-cache guards, frozen eval-mode inference, per-cell atomic
  resume and scalar seals, input/provenance bindings, saved-results API/UI and two
  generated-fixture test files. Preserve training_seed versus bootstrap_seed fields.
  The byte-preservation Git rule is present before new measured reports are created.
- Input-only readiness passed six candidate seals/336 original byte hashes and
  role separation. Preservation preview:234 historical/master/production artifacts.
  No new real-model inference or temperature fitting has occurred at this commit.
- Generated tests:14 passed in 14.06s. Full suite with genuine production/conformal
  integration:536 passed in 100.90s, zero failures/skips. Ruff,Compose configuration and
  whitespace checks pass. Source/config/protocol are committed clean before execution.
  New real scalar/stress outcomes, independent reconstruction and native integration
  remain unverified at this preparation stage. Overall scientific acceptance IN PROGRESS.

## 2026-11-05 — completed matched quality-backbone adaptation v1

- Executed all six runs from clean 2d4243c977a0669776a5dc356d6d6e3166732a12:
  seeds 2026/2027/2028, frozen versus adapted, 15 epochs/795 updates each,
  90 trained epochs/4,770 updates total. Selected epochs frozen 5/6/10 and adapted
  13/8/8 under the declared mean-four-head F1/NLL rule; no early stopping or seed selection.
- Conditional mean four-head macro-F1 rises .50894137 (sample SD .00349332) to
  .54112226 (.00501275). All three descriptive paired primary intervals include
  zero. Primary change +.03433166,95% interval[-.01349020,.06511460]. Retain all27
  contrasts and adverse results: primary clarity/artifact F1 decrease, mean raw
  NLL worsens in every pair, and mean ECE increases for all four heads. Poor-quality
  AUROC falls in all three pairs. Every model misses the one field-class0 image;
  all adapted models have zero artifact-class1 recall on six images.
- Found a paired-row metadata key collision after all learning: bootstrap seed
  20261107 replaced training-seed labels. Preserved the original evaluation.json
  bytes/SHA18e184b5abb4c67a53a17a9887b9b4321f34a904f46e714a925c2b46fdcff188.
  Added evaluation_v1_1.json, explicit correction receipt/script and regression test.
  Only metadata labels/schema change; no fitting, epoch/prediction/metric/interval
  recomputation. Corrected aggregate SHA:
  b21f0305d3d2c19392d91434b7b4dedaed65c9329029b84939143ca8497f2fd8.
- Independent reconstruction passed all90 trained/6 warm metric sets, selected
  ranks, order/flip streams, cosine schedules, optimizer/RNG snapshots and all27
  component intervals. Each adapted backbone changes135 parameter tensors; frozen
  backbones change none. All102 BN buffers stay fixed. All1,008 selected-model
  forward logits reproduce exports bit-for-bit. All1,016 source hashes/cache bytes,
  26 independent geometry samples and171 historical/master/production files match.
  The auditor verifies the exact metadata-only difference and archived execution source.
- Before learning:521 tests passed128.16s. After the metadata-only fix:19 targeted
  tests passed20.53s; full522 passed110.17s, zero failures/skips, genuine production/
  conformal integration. Ruff/Compose/whitespace passed. Source CI42062103457 /
  job127366405116 succeeded; a transient GitHub push rejection succeeded on retry.
- Actual native JPEG/batch/TTA/readiness/simulator/overload checks pass after
  restart. Live-API Streamlit checks verify17 evidence records, all six runs,
  all12 seed/head views, majority/warm references and every class support/recall.
  Both figures visually inspected; corrected plot-only label clipping without
  changing scientific bytes. Docker Linux engine unavailable; no new container/
  browser-rendering claim. Calibration pixels, consumed confirmation/IDRiD, APTOS
  and production artifacts remain preserved. Overall scientific acceptance IN PROGRESS.
- New source: quality_adaptation.py, fixed config/protocol, run/plot/correction
  scripts, two test files, API/UI registry; new aggregate/audit/correction/native/
  verification/figures and results notes. Next: predeclare source-role probability
  calibration and a fixed-model stress comparison; seek independent quality
  validation before any deployment proposal. Formal manuscript drafting stays deferred.

## 2026-11-05 — quality adaptation v1 preparation

- Declared three matched per-seed warm starts, seeds 2026/2027/2028, with frozen
  versus trainable MobileNet backbones and fixed BatchNorm buffers. Exactly 15
  epochs per arm; select by mean four-head macro-F1 then mean unweighted NLL.
  Only 848 fitting/168 selection DeepDRiD images are opened. Calibration metadata
  guard overlap; calibration pixels, consumed confirmation/IDRiD and APTOS are excluded.
- Implemented an atomic epoch state containing optimizer/scheduler/RNG/history,
  all logits and selected snapshot; exact-bound resume and sealed final artifacts.
  Dedicated saved-results API/UI retain all seeds/arms/heads, class supports and
  descriptive paired contrasts. Production remains unchanged.
- Input-only readiness passed. All three genuine warm models reproduced all
  168 selection argmaxes per head; maximum probability delta 2.23638414e-6,
  below predeclared 2e-5. All 1,016 fitting/selection bytes are cache-bound;
  171 historical/master/production artifacts are preservation-previewed.
- Generated fixtures: 18 tests passed 47.32s, including interruption recovery,
  cache-independent RNG, frozen/adapted parameter/BN behavior, role/label guards,
  poor-class0 metrics, component draws and all explorer controls. The initial
  fixture's duplicated missing filenames were corrected before these results.
- Full genuine-model/conformal integration: 521 passed, no failures/skips,128.16s;
  Ruff and Docker Compose configuration passed. One dependency deprecation warning.
  New real-image adaptation training/outcomes and independent reconstruction are
  NOT STARTED at this source commit. Source/config/protocol are committed clean
  before learning. Overall scientific acceptance IN PROGRESS; formal writing deferred.

## 2026-11-04 — completed primary-seed APTOS source optimization v2

- Executed clean a75229f92468a253e8f6c3e28d834571e25bd9e5. The declared primary
  seed 2026 trajectory stopped after 14 epochs/1,862 updates under patience 4;
  selected epoch 10 by macro-F1/NLL. Budget and cosine trajectory both change;
  this single-seed extension establishes neither convergence nor exact-paper reproduction.
- Selection 521: accuracy .80806142,macro-F1 .65156927,QWK .86726292;
  original four-epoch primary .78310940/.63348889. Grade 3 recall falls 17/26 to 7/26;
  grade 4 rises 19/40 to 27/40. Raw NLL .82975311 worsens; source-role temperature
  2.94099575 gives selection NLL .51450645. No deployment selection follows.
- All four descriptive paired component intervals include zero. F1 change versus
  four-epoch primary +.01808038,95% interval[-.04118603,.07879770]; selection/model
  reuse and unknown patient grouping preclude fresh-test or clinical superiority claims.
- All 14 CP comparisons retained. Alpha .10 LAC raw/fallback inclusion .89251440,
  size 1.29558541, grade 3 .46153846/grade 4 .70. Deterministic ordinal raw .90211132 /
  size 2.58349328 versus fallback .98272553/2.66410749. All five randomized means:
  raw .90326296/1.66333973; fallback .94318618/1.70403071, grade 4 only .75.
- Independent reconstruction passed all 14 epoch metrics/rank, schedules/order/flip
  RNG, 211 changed backbone tensors and 147 unchanged BN buffers, temperature objective,
  14,588 raw/fallback sets and 4 paired intervals. All 3,163 source hashes/cache bytes,
  15 independent geometry samples and 134 prior artifacts matched. First epoch's 521
  logits exactly reproduce the original primary before schedules diverge. Report SHA:
  66c49f12bf630493c9dc38180a73daeeae6a222523067f4ec262971b45d2310e.
- Pre-execution 502 tests passed 82.77s. After the complete measured audit, fixed a
  pandas/Torch read-only Boolean-mask warning using owned copies. Actual 262/254
  indices are numerically identical; no scientific fit/inference/report was rerun.
  Post-fix 503 tests passed 91.16s, no failures/skips; Ruff/Compose passed.
- Actual native production JPEG/batch/TTA/readiness/overload and live-API Streamlit
  checks passed after restart: 16 records, all 3 comparators,both alpha/population
  controls, all 5 uniform seeds and single-primary/reuse/stopping captions. Both
  derived figures visually verified. Docker engine unavailable; no new container/
  browser-rendering proof. Reserved APTOS test,consumed IDRiD and production artifacts
  remain preserved. Overall scientific acceptance IN PROGRESS; formal writing deferred.

## 2026-11-04 — primary-seed source schedule extension preparation

- All three previous source fine-tuning runs selected epoch four at their cap.
  Declared an independent 15-epoch cosine trajectory from the original warm start,
  primary seed 2026 only, patience four; all other training settings stay fixed.
  This is source development, not exact-paper reproduction or convergence evidence.
- Added a separate versioned runner/config/protocol and model namespace, immutable
  runtime bindings and exact-provenance resume. The original four-epoch studies,
  consumed IDRiD results, reserved APTOS test and production weights are preserved.
- Targeted generated-fixture tests: 58 passed, 59.50s; Ruff passed. Actual input-only
  readiness verified train/selection/calibration 2115/521/527 with no model fitting
  or inference. Full suite: 502 passed, no failures/skips,82.77s with genuine
  production-model/conformal integration; Ruff and Compose configuration passed.
  The tested source/config/protocol are committed clean before actual learning.
- Saved-results API/dashboard support is implemented; actual new retinal training,
  independent reconstruction and native integration remain unverified at this stage.

## 2026-11-04 — first locked IDRiD external outcomes consumed

- Execution from clean e447a21b55b82ca47d1e3b91caf76bca61f23699 completed all 4
  frozen classifiers on 516 images/2,064 unique model-observations, all 56 conformal
  comparisons and nine paired full/filtered comparisons. Prior temperatures and
  all score thresholds were reused unchanged; no fitting, deployment switch or
  APTOS reserved-test inference occurred. IDRiD outcomes are now consumed.
- Primary 2026/main 505: accuracy .42970297, macro-F1 .41359165, QWK .65164044,
  source-T NLL 1.38231498; grade-3 recall  40/91=.43956044,grade-4 recall 26/60=.43333333.
  Baseline accuracy .43366337/F1 .36194750; other fine-tuning F1 values .40590214
  and .35787819. Conditional fine-tuning mean F1 .39245733 / sample SD .03019221.
  No favorable seed or population was substituted for the fixed primary.
- Primary alpha .10 LAC inclusion .70693069/size 1.93069307; deterministic ordinal
  .96039604/3.46138614. Five randomized raw replicates mean .83287129/2.53900990;
  common fallback .83801980/2.54455446. All56 rows/populations/grade denominators
  are retained. Wider sets are not a new triage-utility or coverage guarantee.
- Primary paired main 505 F1 difference +.05164415, descriptive 95% interval
  [.00889026,.09467970]; accuracy difference -.00396040, interval [-.04950495,
  .03960396]. No multiplicity, patient-independence or clinical superiority claim.
  Main annotation sensitivity accuracy .42772277–.43366337 is hypothetical under
  observed component-label alternatives, not corrected truth or a confidence bound.
- Independent math reconstruction passed all 24 raw/calibrated grading views,
  56 comparisons/57,792 sets, 360 annotation envelopes, 18 paired bootstrap intervals,
  conditional summaries and primary/consumed flags. All516 source hashes/cache
  bytes matched; 25 canonical images were independently reconstructed. All 31 prior
  master/model/results/audit artifacts matched. Evaluation SHA256:
  812c82d7771752fa00d076e067e6d1692da4a414c0b1207c8e6df2f21ec35282.
- Full 491-test pre-execution suite passed 84.64s, zero failures/skips; Ruff and
  Compose configuration passed. Real native JPEG/batch/TTA/readiness/simulator
  and live-API Streamlit component checks passed after restart: 15 records, 4
  classifiers, 3 population controls, all 5 uniform draws and retained fixed-primary/
  consumed captions. Original production card/sidecar remain matched. Both derived
  figures were visually inspected; plot-only labels shortened without changing
  scientific bytes. New browser/container execution is unverified; Docker unavailable.
- No external-quality truth, useful validated triage or exact-paper reproduction
  is established. Future development must use source development data; disclose
  any IDRiD reuse and never tune final models/policies to these outcomes. Overall
  scientific acceptance IN PROGRESS; formal manuscript drafting remains deferred.

## 2026-11-04 — declared locked IDRiD external-domain evaluation

- Fixed four existing EfficientNet-B0 states: frozen-feature baseline and all3
  fine-tuning seeds, with primary2026 and main505 representatives unchanged.
  Full516/filtered513 remain mandatory sensitivity views. No model outcome from
  IDRiD has yet been observed; all prior source/overlap/label findings are disclosed.
- Bound all source/model/population/cache inputs. Dedicated external cache accepts
  only training-ineligible external_holdout; prediction is eval-only and state-hash
  checked. Ten source-development images per model provide regression preflight.
- Reuse every existing APTOS temperature and all56 CP artifacts without fitting.
  External uniforms use child2 of a three-child SeedSequence; previous source
  calibration/selection streams remain separate. Annotation envelopes and paired
  fixed-component bootstrap are predeclared, with ambiguity/independence limits.
- Generated scoring/cache/artifact tests:12 passed,13.35s. Ruff found an unused
  import, now removed. Full API/UI suite and clean source commit precede execution.
- Real default readiness passed with no external inference. Full pre-execution
  suite:491 passed, zero failures/skips,84.64s with genuine production-model/CP
  integration; Ruff and Compose configuration passed. An intermediate490-test
  suite passed78.65s before adding exact-decimal score export/replay regression.
  All sources/config/protocol are committed clean before the first external pass.

## 2026-11-04 — completed IDRiD source/pixel review and population freeze

- Executed from clean d1c5968189f8bd36508b5cbb4c35220af5f0bf6b. Source origin is
  now user-confirmed in a separate bound receipt; original data audit SHA remains
  b31891fc14c5c52ab1e638821a5e014762e20d2add6a936ab18d093ef165f88a.
  This is not official archive-checksum authentication or clinical adjudication.
- Reviewed all 8 internal/all 3 cross-source pairs. Six internal pairs are byte-
  identical; three of those have different DR labels. The other 2 internal and all 3
  cross-source pairs remain unresolved fingerprint candidates. No cross-source
  pair has identical original RGB or canonical 224px arrays. Source grades stay intact.
- Froze three label-independent sensitivity views: full 516 observations/508
  components; conservative cross-source-filtered 513/505; one representative per
  retained component 505/505. All four grade-conflicting components remain flagged
  in each view. Masks use original candidate components and fixed identity hashes,
  without consulting labels, pixel similarity, predictions or performance.
- Independent reconstruction reproduced all 11 pixel comparisons and all 516
  membership decisions, original columns/grades, hashes and source receipt.
  All 20 previously preserved master/production/APTOS artifacts and the frozen
  IDRiD audit/private inputs matched. Review SHA256:
  f21a3cd6109030c6c9da96710b7af0e6a89a00365afbfc76af9cb539abdf15f7.
- Full 476-test suite passed 77.12s, zero failures/skips with genuine-model/CP
  integration; Ruff and Compose configuration passed. Actual native JPEG single/
  batch/TTA/readiness/simulator and live-API Streamlit component checks passed
  after restart. Fourteen evidence records show current source confirmation,
  historical audit status, retained overlap warning and all frozen population
  counts. Original production card/sidecar match; no external model prediction.
- New container/browser rendering remains unverified; Docker engine unavailable.
  Clinical-grade ambiguities and cross-source perceptual candidates remain open.
  Next declare ambiguity-sensitive external analysis with locked models and full/
  filtered populations. Overall project IN PROGRESS; formal writing deferred.

## 2026-11-04 — IDRiD source confirmed and pair-review protocol declared

- User answered "Yes, official IDRiD/IEEE DataPort release". Recorded a separate
  receipt bound to the frozen report/manifest/image-content hashes. Original audit
  bytes and its execution-time unconfirmed status remain unchanged; source
  testimony is not represented as official archive-checksum authentication.
- Added fail-closed current-provenance resolution for API/UI. Missing, stale or
  malformed receipts cannot upgrade the status or clear the overlap gate.
- Declared all11 fixed candidate-pair comparisons and three label-independent
  sensitivity populations before real pair review. Pixel proof distinguishes
  source bytes, original displayed RGB content and descriptive canonical geometry.
  No original grade is corrected; no external model inference occurs.
- New generated software tests:16 passed,1.54s. Removed an unused test import
  found by Ruff. Full updated suite and source commit precede real execution.
- Final pre-execution suite:476 passed, zero failures/skips,77.12s with genuine
  production-model/conformal integration enabled; Ruff and Compose configuration
  passed. Receipt/overlap, row-order/label independence and pixel-evidence tests pass.

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
  genuine frozen-ImageNet-feature candidates and all 14 CP comparisons in1435.90s.
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
  states, direct head×feature probabilities and all 14,588raw/fallback sets.
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
- Implemented two frozen official ImageNet candidate runs and all 14 declared
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
