# QSART-X panel demonstration

Allow five minutes after setup. Show the local screening workflow and two
retrospective DeepDRiD studies. [Release setup and verification](QSART_X_RELEASE.md).

## Preflight

1. Start Docker Desktop and the Compose stack with the environment files and
   project name used for the rehearsal. `docker compose ps` with those same
   options lists the published bindings. For a new stack, follow the
   [release setup](QSART_X_RELEASE.md#start-the-release).
2. Open the four services using the active port bindings. All URLs use
   `http://127.0.0.1`.

   | Service and route | Example setup | Verified `qsart-x-verify` mapping |
   | --- | ---: | ---: |
   | Public-facing site `/demo`, `/research`, `/publications` | `18700` | `18727` |
   | Authenticated screening workspace `/` | `18701` | `18701` |
   | Research lab `/` | `18501` | `18627` |
   | FastAPI `/health`, `/ready`, `/portal/ready` | `18000` | `18127` |

3. Run the [preflight command](#readiness-command). A `ready` result confirms
   service readiness, the release model metadata, and both audited study
   results served through the API. Open **Experiment Explorer** for the walkthrough
   and keep the cached charts below available for offline use.
4. Prepare local `research_operator` and `clinician_reviewer` accounts using
   the [account setup](../README.md#analyze-an-image-in-the-local-workspace).
   Sign in before presenting or switch roles outside the shared screen. Use
   a rehearsal workspace with demo cases. Choose a JPG or PNG of up to
   12 MiB cleared for research and panel display, or use the synthetic fixture
   below. Label the synthetic fixture **DEMO DATA** when presenting it.

### Readiness command

Run from the repository root in the project Python environment. The default
ports match the example Compose setup.

```powershell
python -m scripts.preflight_qsart_x_demo
```

For the verified `qsart-x-verify` mapping, pass its three different origins.
Use `--workspace` too if your workspace listens on a different port.

```powershell
python -m scripts.preflight_qsart_x_demo --api http://127.0.0.1:18127 --public http://127.0.0.1:18727 --lab http://127.0.0.1:18627
```

The command uses local GET requests and the checked-in release reports. It
returns exit code `0` for `ready` and `1` for `attention_required`, with one
result per check. It uses no accounts or images. Sign in separately to rehearse
the operator and reviewer steps.

### Synthetic demonstration image

To prepare the synthetic fixture, run from the repository root with the project
Python environment active. The source illustration is included in the site.

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
python -c "from PIL import Image; Image.open('apps/web/public/retina-illustration.webp').convert('RGB').save('artifacts/DEMO_DATA_fundus.png')"
```

## Five-minute walkthrough

| Time | Screen | What to show |
| --- | --- | --- |
| 0:00–0:45 | Public **Research** and **Publications** | Introduce separate specialist-review and recapture capacity limits. Point to the two study reports and the unpublished manuscript for author review. |
| 0:45–2:00 | Operator workspace | Choose **New case**, **Choose image**, then **Analyze and save case**. Show the grade distribution, image quality, uncertainty, and model recommendation. One image illustrates the workflow. Capacity limits apply to cohorts. |
| 2:00–3:00 | Reviewer workspace | Open the rehearsal case in **Case queue** under **All cases**. In **Reviewer assessments**, select an **Assessed grade**. To show an override with synthetic data, select a different **Override action** and enter a reason in **Review notes**. Click **Save assessment** and show the assessment alongside the original model result. The private JSON and print view are available from the case screen. |
| 3:00–4:00 | Lab **Experiment Explorer**, `set_c_validation_v1` | Keep **Set-C policy mode** on **Deployed ordinal**. Review caps of 30%, 60%, and 90% overloaded. With a 100% image review cap, all 99 patients received a specialist-review recommendation. All 12 policy comparisons with matching image and patient workloads made identical recommendations. |
| 4:00–5:00 | Lab **Experiment Explorer**, `paired_view_v1` | Compare **Extra-view budget** 0% and 10%. Missed referable eyes fell from eight to six with eight additional fields on 84 development eyes. At 10%, select **Research cost per eye** under **Matched acquisition workload**. All 16 matched cost intervals across the study included zero. |

The workspace runs trained inference and saves a research record. A synthetic
case demonstrates software behavior. The lab controls display saved aggregate
analyses. Set-C was held out within the same DeepDRiD release. The acquisition
study used recorded same-eye fields from the development cohort. New capture,
retake, and clinical follow-up outcomes remain unobserved.

## Offline fallback

If live inference is unavailable, identify the demo as an offline review and
walk through the saved figures and tables. Keep the local report files open so
the panel can inspect the denominators without network access.

- [Set-C patient dispositions figure](../reports/set_c_validation_v1/patient_dispositions.png)
  and [study results](research/SET_C_VALIDATION_V1_RESULTS.md).
- [Paired-view workload figure](../reports/paired_view_v1/acquisition_workload.png)
  and [study results](research/PAIRED_VIEW_V1_RESULTS.md).
- [Manuscript evidence table](research/QSART_X_EVIDENCE_TABLE.md) and
  [release verification receipt](../reports/qsart_x_release_v1/verification.json).
