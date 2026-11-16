# QSART-X local research release

The release packages public information, authenticated screening, persistent
case review, and the research lab in the CPU Docker Compose overlay. Use the
local workspace with research-authorized images whose pixels are clear of
identifiers. The studied outcomes are retrospective. In the locked Set-C
study, the deployed ordinal policy needed full review capacity. The paired-view
study's matched cost intervals all included zero. [Set-C results](research/SET_C_VALIDATION_V1_RESULTS.md) ·
[Paired-view results](research/PAIRED_VIEW_V1_RESULTS.md).

## Start the release

Use the PowerShell steps in the README to create the private `.env.qsart-x`
file from the example and put the matching trained artifacts under the local
checkpoint mount. Run the following from the repository root.

```powershell
docker compose --env-file .env.qsart-x -f docker-compose.yml -f infrastructure/docker/compose.qsart-x.yml up --build -d --wait
```

| Local port | Service |
| ---: | --- |
| 18700 | Public information and model readiness |
| 18701 | Authenticated screening and review workspace |
| 18501 | Research lab and Experiment Explorer |
| 18000 | FastAPI and OpenAPI docs at `/docs` |

The migration service runs before the API. The database stays on the internal
Compose network. Back up the `portal-data` and `portal-images` volumes together
for a project with saved cases. Use the same environment files and Compose
project when provisioning accounts. [Startup and account setup](../README.md#qsart-x-platform-setup).

`/health` checks API liveness, `/ready` checks trained artifacts, and
`/portal/ready` checks the database schema. If the model is unavailable, check
the configured checkpoint and calibration paths and restart the API. If the
database is unavailable after a restart, check the `db` health and `migrate`
exit status before attempting account setup. [Dataset and model evidence](QSART_X_DATASETS.md).

## Verify the running stack

Run the software suite and check the generated manuscript tables.

```powershell
python -m pytest -q
python -m ruff check .
python scripts/export_qsart_x_tables.py --check
```

Run `scripts/verify_deployment.py` with the bound real pilot manifest, the
`validation` split, the local API and lab URLs, and `--require-conformal`.
This exercises real single-image and batch inference, model readiness, and
the separate review and recapture capacity checks.

Use `scripts/verify_qsart_x_release.py` against a disposable Compose project
to verify persistence and authorization. Pass the `--workspace`, `--api`,
`--public` and `--lab` loopback origins and a synthetic image via `--image`.
Provide a JSON object on standard input with `operator`, `reviewer`, `viewer`,
and `other_operator` entries, each containing a disposable account's `username`
and `password`. With `--write-witness`, save the synthetic case witness under
ignored `artifacts/`. Stop and restart the disposable project with its volumes,
then run the verifier using `--replay-witness` in place of `--image` and
`--write-witness`. The optional `--output` receipt contains aggregate check
statuses. [Release verification](../reports/qsart_x_release_v1/verification.json).

## CPU profiles

Set `QSART_CONTAINER_CPU_THREADS` in the private Compose environment file to
select the inference thread count, then rerun Compose to recreate the API.
The release benchmark used matching 1-, 2- and 4-thread containers, the same
12 verified DeepDRiD development images, and three rotated repeats.

| Threads | First-view request p50 / p95 | Forced-flip request p50 | Four-photo batch request p50 |
| ---: | ---: | ---: | ---: |
| 1 | 184.1 / 363.1 ms | 198.3 ms | 758.0 ms |
| 2 | 181.8 / 387.3 ms | 192.1 ms | 747.8 ms |
| 4 | 181.3 / 381.0 ms | 193.6 ms | 754.8 ms |

Grades, actions, conformal sets, and extra-pass choices agreed across all
three configurations. Numerical outputs met the measured tolerance. The
Compose default uses two threads. The small loopback sample is specific to
the tested host and workload. The benchmark includes peak process and
container memory.
[CPU benchmark](../reports/qsart_x_release_v1/cpu_profiles.json).

## Manuscript materials

Use the [manuscript draft](research/QSART_X_MANUSCRIPT.md) and the
[evidence table](research/QSART_X_EVIDENCE_TABLE.md) for exact outcomes,
denominators, and audit links. The [IEEE LaTeX export](research/QSART_X_TABLES.tex)
contains both tables and the figure includes. Use `graphicx` and add
`\input{docs/research/QSART_X_TABLES.tex}` to the manuscript when compiling
from the repository root. The historical draft documents the original QSART
experiments. [Verified literature matrix](LITERATURE_MATRIX.md) ·
[Bibliography](research/references.bib) · [Historical draft](research/MANUSCRIPT_DRAFT.md).
