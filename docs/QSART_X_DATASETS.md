# QSART-X dataset imports and split boundaries

Open **Dataset Catalog** in the local Streamlit research lab, or read
`GET /research/datasets` from the local FastAPI service. The catalog shows
aggregate label distributions, source access, QSART split roles, and the SHA-256
digests of the reports behind each entry. It reads the mounted `reports`
directory. An unavailable entry means the required aggregate evidence is
missing or does not match its recorded audit. The catalog works without local
photos or trained model files.

## Recorded cohorts

| Dataset | Imported and checked | QSART use |
| --- | --- | --- |
| APTOS 2019 | 3,662 labeled PNGs. 148 excluded through whole-component conflict quarantine. | 2,115 fitting, 521 selection, 527 calibration, and 351 reserved internal-test photos. The separate competition test was outside the import. |
| DeepDRiD regular fundus | 1,184 Set-A development photos, 340 Set-B confirmation photos, and 400 Set-C photos. | Set-A roles use patient-disjoint groups. Set-B confirmation outcomes are consumed. Set-C remains unscored. Its locked overlap screen leaves 396 eligible photos from 99 patients. |
| IDRiD grading | 516 photos, including 413 and 103 in the two original source splits. | Both source splits are external to QSART. Its 516, 513 and 505-photo views are overlapping sensitivity populations with previously observed outcomes. |
| EyeQ | 28,792 author annotations. The transformed mirror matches 12,529 of 12,543 author-train annotations and zero of 16,249 author-test annotations. | Metadata readiness only. Original photos, cross-source overlap, and a quality harmonization protocol remain pending. |

APTOS temperature fitting and conformal calibration use disjoint fingerprint
groups, with 262 and 265 photos respectively. Fingerprint components in APTOS
and IDRiD capture detectable related images. Patient identity is unavailable.
DeepDRiD provides patient and eye grouping. Keep both views of each eye together
when a protocol uses all four photos. In IDRiD, macular edema risk remains a
separate source label. EyeQ retains the Good, Usable and Reject categories.
[APTOS audit](research/APTOS_V1_RESULTS.md) ·
[DeepDRiD data contract](DATA_SCHEMA.md) ·
[Set-C audit](research/DEEPDRID_ONLINE_READINESS_V1_RESULTS.md) ·
[IDRiD results](research/IDRID_EXTERNAL_V1_RESULTS.md) ·
[EyeQ cache coverage](research/EYEPACS_HF_V1_RESULTS.md).

## Recheck the existing imports

Run this offline from the QSART root using the local Python environment.

```powershell
.\.venv\Scripts\python.exe scripts/verify_dataset_imports.py --include-eyeq
```

The checker verifies frozen manifest and witness digests, original photo
SHA-256 values, and split membership, then prints aggregate counts. Use
`--aptos-root`, `--idrid-root` and `--deepdrid-author-root` to override source
locations. APTOS and IDRiD accept their existing local environment settings.
DeepDRiD Set-A/B photo paths come from the frozen manifests. `--include-eyeq`
checks the seven pinned author metadata files and their annotation profile.
It requires those local files and skips original EyePACS pixels.

The command checks this project's historical import, including the exact
recorded CSV and witness bytes. New imports may contain different local path
strings. Keep their manifests and receipts versioned in a separate private
output directory. For storage and provenance, keep original source archives,
terms receipts, image paths, images and row-level manifests outside Git. The
app serves aggregate counts and report digests.

## Acquire or import locally

### APTOS 2019

Download the labeled competition export from Kaggle, following the account
access and terms presented there. The importer reads `train.csv` and the
original `train_images` PNGs. For a separate local reproduction, choose fresh
output paths and retain the published audit. [Official competition files](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) ·
[APTOS preparation and relocation](APTOS_SETUP.md).

```powershell
.\.venv\Scripts\python.exe scripts/prepare_aptos_archive.py `
  --archive 'PATH_TO_OFFICIAL_APTOS_ZIP' --conflict-policy quarantine `
  --raw-root data/raw/aptos_reproduction --output data/manifests/aptos_reproduction `
  --private artifacts/aptos_reproduction --audit reports/private/aptos_reproduction.json
```

Use the development-only manifest for fitting and selection. The master split
table records the reserved internal test so that its group membership can be
verified. The source archive and all labels are kept as released.

### DeepDRiD

Use the pinned author repository and regular fundus folders. `prepare_data.py`
supports the bounded official pilot download or an existing local root plus
the source training and validation CSVs. The development preparation script
builds the patient-aware cohorts from that pilot and the pinned metadata.
The Set-C guide reproduces the authenticated inventory and patient-level
overlap screen with the author package and the pinned Git tree. Retain the
source attribution and license with the local package.
[Pinned DeepDRiD source](https://github.com/deepdrdoc/DeepDRiD/tree/56d8af71319803150d0e7b90533e0918c7bf85ee) ·
[Manifest schema and commands](DATA_SCHEMA.md) ·
[Set-C reproduction](research/DEEPDRID_ONLINE_READINESS_V1_RESULTS.md#reproduce).

### IDRiD

Open the official challenge or IEEE DataPort route and obtain the **disease
grading** package under the supplied terms. Select the folder that holds
`1. Original Images`, `2. Groundtruths` and the license files. The readiness
auditor binds those files and screens the two source partitions against frozen
APTOS and DeepDRiD references. The catalog reflects the recorded user source
confirmation and the later consumed evaluation. An official archive checksum
was unavailable in that receipt.
[Official challenge data](https://idrid.grand-challenge.org/Data/) ·
[IEEE DataPort record](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) ·
[IDRiD reproduction protocol](research/IDRID_DATA_V1_PROTOCOL.md) ·
[Source confirmation and review](research/IDRID_REVIEW_V1_RESULTS.md).

### EyeQ

The metadata fetch downloads seven small pinned author files. For source
photographs, obtain the original JPEGs through official EyePACS access and keep
the applicable terms receipt. Follow the original-image inventory instructions
before considering evaluation. The transformed mirror has a separately audited
metadata join and incomplete split coverage.
[Official EyeQ source](https://github.com/HzFu/EyeQ) ·
[EyePACS competition data](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) ·
[EyeQ setup](EYEQ_SETUP.md).

```powershell
.\.venv\Scripts\python.exe scripts/audit_eyeq_data.py --fetch-metadata
```

The [data access audit](research/DATA_ACCESS_AND_TERMS.md) records each source's
research scope and terms. The [Experiment Explorer](SPEC_V1_STATUS.md) also
documents the supporting FDR26 and MSHF studies.
