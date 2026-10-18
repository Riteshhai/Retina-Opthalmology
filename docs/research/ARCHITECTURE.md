# QSART implemented architecture

This describes the current source and completed real-image pilot, 18 October 2026. Diagrams show implemented paths; they do not imply predictive quality, clinical authorization or verified container deployment.

## Data and training path

```mermaid
flowchart TD
    S["Official DeepDRiD pinned revision<br/>Photos, labels, definitions and license"] --> P["prepare_data.py<br/>Parse eye-specific grades and quality codes<br/>Quarantine ambiguous source rows"]
    P --> V["validate_data.py<br/>Decode, checksum, duplicate and grouping audit"]
    V --> M["Frozen local manifest<br/>260 images from 65 patients"]
    M --> T["Fitting<br/>160 images / 40 patients"]
    M --> W["Checkpoint selection<br/>20 images / 5 patients"]
    M --> C["Temperature fitting<br/>20 images / 5 patients"]
    M --> H["Official-validation pilot holdout<br/>60 images / 15 patients"]
    T --> DR["train_dr.py<br/>MobileNetV3-small<br/>5 DR grades"]
    T --> Q["train_quality.py<br/>Separate MobileNetV3-small<br/>2 overall-quality classes"]
    W --> DR
    W --> Q
    DR --> K["Validation-loss-selected checkpoints<br/>DR epoch 1 and quality epoch 2"]
    Q --> K
    K --> TC["calibrate.py<br/>Original and flip-mean temperatures<br/>Checksums and calibration-fit diagnostics"]
    C --> TC
    K --> E["evaluate.py<br/>Frozen predictions and cohort allocations"]
    TC --> E
    H --> E
    E --> R["Aggregate reports<br/>Grading, quality, 7 policies, ablations<br/>Patient-cluster interval and figures"]
```

Shared preprocessing crops the fundus, letterboxes to a square, converts RGB and applies recorded normalization. The pilot uses 96px and three CPU training epochs from random initialization. Natural defect codes are retained but the current quality checkpoint learns only overall quality. All known views and both eyes of a supplied patient stay together. Final official evaluation data remain untouched. Raw photos, manifests, checkpoints and individual predictions stay local and outside Git.

## Artifact-backed request path

```mermaid
flowchart TD
    U["Streamlit dashboard<br/>Default local API connection"] --> A["FastAPI upload endpoints<br/>PNG/JPEG, size, decode and dimension checks"]
    A --> READY{"Required validated artifacts loaded?"}
    READY -- "No" --> UN["HTTP 503<br/>No substituted prediction"]
    READY -- "Yes" --> PRE["Shared preprocessing"]
    PRE --> F["CPU DR and overall-quality forwards<br/>Temperature-scaled probabilities"]
    F --> B{"Extra pass selected<br/>within floor(N times compute budget)?"}
    B -- "Yes" --> FL["Horizontal flip for both models<br/>Mean logits and flip-specific temperature"]
    B -- "No" --> O["Grade probabilities and referable probability<br/>Poor-quality probability and entropy"]
    FL --> O
    O --> MODE{"Request scope"}
    MODE -- "Single" --> PT["Pointwise rule<br/>Severity first, quality, uncertainty"]
    MODE -- "Batch" --> LP["Three-action assignment LP<br/>Separate integer review and recapture capacities"]
    LP --> FEAS{"Feasible quality deferrals?"}
    FEAS -- "No" --> ERR["HTTP 422<br/>Capacity infeasibility reported"]
    FEAS -- "Yes" --> OUT["One action per image<br/>AUTO_SCREEN / RECAPTURE / SPECIALIST_REVIEW"]
    PT --> OUT
    OUT --> UI["Structured JSON with reason codes and timing<br/>Dashboard tables and CSV/JSON export"]
```

`InferenceEngine` rejects missing, untrained, malformed, nonfinite or software-test artifacts in production loading and checks task, manifest/split provenance and calibration-checkpoint hashes. `/health` reports the API process; `/ready` reports artifact availability. Neither endpoint certifies diagnostic accuracy. The real pilot loads trained artifacts despite poor predictive performance, so the model status and research warning remain essential context.

The selected second pass is horizontal-flip mean logits, not a separate deeper model. Its cap applies to the current request population; the default single-image call receives no extra pass unless explicitly enabled. `/predict` has no population triage-budget guarantee. `/predict/batch` uses the supplied review/recapture capacities and reports their realized action rates; infeasibility is explicit. `AUTO_SCREEN` carries a provisional grade and can be positive. Deferred actions do not imply a completed diagnosis or improved recapture.

## Source map and runtime packaging

| Responsibility | Implemented source |
| --- | --- |
| Metadata and manifest construction | `qsart/deepdrid.py`, `qsart/data/manifests.py`, `scripts/prepare_data.py` |
| Shared pixel transformation | `qsart/preprocessing/fundus.py`, `qsart/training/data.py` |
| Network and checkpoint validation | `qsart/models/networks.py`, `qsart/models/checkpoint.py` |
| Training and calibration | `qsart/training/trainer.py`, `qsart/training/cli.py`, `qsart/calibration/temperature.py` |
| Inference and compute control | `qsart/inference/engine.py` |
| Pointwise and cohort allocation | `qsart/triage.py` |
| Grading, policy and interval reporting | `qsart/evaluation.py`, `scripts/evaluate.py`, `scripts/benchmark_cpu.py` |
| HTTP service and response schema | `api/main.py`, `api/schemas.py` |
| Dashboard | `dashboard/app.py` |
| Container definition | `Dockerfile`, `docker-compose.yml` |

Compose defines separate API and dashboard services using the same image, loopback-bound host ports 8000 and 8501, read-only checkpoint/report mounts and writable temporary filesystems. The dashboard calls the API service. The first image built successfully; latest-source rebuild and real-artifact container checks were pending at this documentation update. Container deployment is unverified until those checks complete.

See [model status](../MODEL_STATUS.md), [data status](../DATA_STATUS.md), [the Review-1 briefing](REVIEW1_BRIEF.md) and the [pilot report](../../reports/pilot/evaluation.json) for measured results and limitations. The held-out DR model predicts grade 3 for all 60 inputs; diagram completeness must not be interpreted as a clinically useful classifier.
