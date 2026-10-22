# QSART change log

## MASTER_SPEC_v1.0 adoption — 2026-10-20

- Adopted the user's updated master specification; exact root/versioned copies
  and SHA recorded in docs/SPEC_ALIGNMENT.md. Preserved all existing work.
- CR-001: changed current automatic outcome from provisional AUTO_SCREEN to
  AUTO_NONREFERABLE. New versioned triage enforces referability/quality/uncertainty
  eligibility and explicit overload. Affects API schemas, routing, simulator,
  dashboard, metrics and end-to-end tests. Old v1 research results keep old semantics.
- CR-002: add conventional and paper-inspired ordinal conformal outputs through
  a checksum-bound sidecar with separate temperature/conformal calibration groups.
  Affects uncertainty, API, dashboard and real calibration tests. No clinical,
  conditional-routing or external-shift coverage guarantee is implied.
- New dataset/model direction: APTOS development and EfficientNet-B0 baseline,
  subject to official access and genuine training; keep current working DeepDRiD
  MobileNet deployment. External IDRiD remains evaluation-only.
- Added simulator/research endpoints, comparison/explorer backlog and full-chain
  verification requirements. Formal manuscript writing follows main experiments;
  existing draft is preserved. Overall research acceptance remains partial.

Required checks: eligibility and zero/overload budgets; conformal quantiles,
ordering, bindings and partition separation; API/client compatibility; real JPEG
single/batch/sidecar/simulator; Ruff/pytest; rebuilt CPU Docker and restart.

## Alignment verification — 2026-10-21

198 integrated tests passed with real models/conformal; Ruff and Compose passed.
CPU Docker workflow and restart were exercised; 37 runtime source files matched.
EfficientNet trained on DeepDRiD but did not win the selection macro-F1 rule.
Conservative ordinal sets were vacuous; all feasible grid cases used full review.
These unfavorable research findings are preserved. APTOS is not downloaded.
Overall scientific acceptance remains IN PROGRESS; see docs/SPEC_V1_STATUS.md.

## Controlled degradation development study — 2026-10-22

- Predeclared and executed seven fixed 224px model-input conditions on all 168
  selection images / 42 patients. Clean predictions exactly match ordinary
  inference, with one geometric preprocessing pass. No parameters were retuned.
- Added descriptive paired patient-cluster intervals, quality/severity strata,
  full grading/calibration/CP metrics and aggregate plots. Strong blur and JPEG
  reduced accuracy; JPEG did not raise the mean poor-quality score. Ordinal sets
  remain vacuous. Confirmation/test/external data remain unused in this study.
- Added aggregate stress results to research discovery for the dashboard explorer.
- Fixed a decimal-alpha conformal rank boundary with grouped/image regressions;
  current alpha .10/.05 sidecar bytes and predictions are unchanged.
- Docker Desktop failed before the engine started because of a stale socket.
  Native PowerShell services passed real HTTP/CP/simulator checks on 18000/18501.
  Current Docker execution is pending; historical 21 October proof is preserved.
