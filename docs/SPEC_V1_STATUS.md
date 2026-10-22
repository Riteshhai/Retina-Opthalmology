# QSART Development Status — updated 22 October 2026

- **Current specification:** MASTER_SPEC_v1.0, exact root/versioned copies retained.
- **Verified pre-adoption commit:** cedf182846845a782bb6e30336d9c8f99b2e31fb;
  new implementation source and aggregate evidence are delivered on the same
  qsart/research-prototype branch/draft PR. Exact v1 source is preserved locally.
- **Current milestone:** specification alignment and integrated real-image
  execution VERIFIED COMPLETE; overall scientific project IN PROGRESS.
- **New execution source:** `493ad7c6e37d16dc338e00280d8ac714ee23a0c8`;
  fixed seven-condition selection-data degradation VERIFIED COMPLETE. Full
  [results and limitations](research/ROBUSTNESS_V1_RESULTS.md) retain the observed
  blur/compression failures and non-monotonic quality response.

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
duplicates, without fabricating patient/eye/quality labels. APTOS is not acquired.

**Verification:**198 tests passed, no failures/skips, 78.51 seconds. This includes
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

**BLOCKED:** official APTOS export/access. The user confirmed it is not downloaded.
[Acquisition/local preparation](APTOS_SETUP.md) documents the exact next step.
APTOS and independent IDRiD results, clinical review/recapture outcomes and
prospective utility have not been produced.

**IN PROGRESS overall:** useful conformal/triage behavior, independently tuned
comparators, repeated seeds, additional robustness/ablations and external-domain
evaluation. A future experiment must predeclare changes; consumed v1 confirmation
cannot become an undisclosed tuning set. Formal manuscript writing stays deferred.

Key files: MASTER_SPEC.md, CHANGELOG.md, AGENTS.md, TASKS.md, PROGRESS.md,
qsart/triage_v2.py, qsart/uncertainty/conformal.py, conformal sidecar, APTOS adapter,
EfficientNet training support, api/research.py, schemas/main, eight-page dashboard,
reproducible scripts, new tests and reports/spec_v1. Photos, patient manifests,
raw per-image scores, downloaded author code and checkpoints remain local/ignored.

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
real-image/model/conformal/batch/simulator HTTP checks. A current container run is
**BLOCKED** by that host startup failure. Historical Docker evidence is preserved.

22 October verification: **223 passed**, no failures/skips, 67.03 seconds with
genuine-model/sidecar integration enabled; Ruff and Compose configuration passed.
The added real development experiment has exact clean-equivalence checks on all
168 images. Independent reconstruction verified all 1,176 LAC sets, all seven
conditions' metrics, six paired accuracy intervals and the recorded hash bindings.
The native HTTP workflow passed again after restarting the launcher, and the new
experiment was retrieved through the actual research endpoint. Evidence is in
reports/robustness_v1; the previously measured 198-test/container results remain
historical records rather than being overwritten.
