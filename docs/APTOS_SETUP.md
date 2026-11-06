# APTOS 2019 acquisition and local preparation

Status on 3 November 2026: **VERIFIED COMPLETE — source validation and partitions**.
The user confirmed downloading the APTOS 2019 competition archive from Kaggle.
All 3,662 labeled training images passed archive/decode/hash checks. Under the
declared quarantine rule, 148 images in 51 conflicting-label fingerprint
components are excluded without relabeling. The 3,514 retained images are split
into 2,115 training, 521 selection, 527 calibration and 351 reserved test images.
The [development classifier/uncertainty study](research/APTOS_V1_RESULTS.md) is
VERIFIED COMPLETE, including independent reconstruction. Preparation follows
[APTOS_V1_PROTOCOL.md](research/APTOS_V1_PROTOCOL.md). The earlier 21 October
missing-data blocker is superseded. No credentials need to be shared.

On 4 November the user relocated the source into the major-project workspace.
The annotation and all 3,662 training PNG hashes match the original validated
source. The ignored local .env now supplies QSART_APTOS_ROOT; the inspector reads
this setting automatically. Process environment takes precedence over .env, and
an explicit --aptos-root takes precedence over both. Relative configured values
resolve from the QSART directory. Frozen manifests, feature caches and measured
results retain their original bindings; relocating a source does not rerun or
repartition the completed experiment.

```powershell
.\.venv\Scripts\python.exe scripts/inspect_datasets.py --output artifacts/aptos_v1/current_inventory.json
```

This command inventories the configured source. It does not overwrite the frozen
study or perform reserved-test inference. The independent relocation checksum
proof is reports/aptos_v1/relocation_verification.json.

Obtain the labeled training export from the
[official competition data page](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
using your account and the applicable competition access steps. Complete any
required agreement yourself. Do not use an unverified mirror or share credentials
in chat. Keep original downloads and images inside the ignored local data tree.

Expected local structure:

```text
data/raw/aptos2019/
  train.csv                 # actual id_code,diagnosis fields
  train_images/
    <actual_id_code>.png
```

To import the original official archive safely, from the QSART project directory:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_aptos_archive.py --archive "<absolute path to aptos2019-blindness-detection.zip>" --conflict-policy quarantine
```

Only train.csv and its training PNGs are extracted into the ignored project tree.
The original ZIP stays unchanged. The importer verifies paths, source schema,
CRC, original hashes, decodability and duplicate fingerprints. This study's
explicit quarantine policy excludes every image in a label-conflicting component
without relabeling; strict rejection remains the adapter/importer's default.
The private manifest, exclusions and calibration-role map are frozen under
data/manifests/aptos_v1; public evidence contains aggregates and hashes only.

For an already extracted source and strict conflict rejection, the older adapter
command remains available:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_datasets.py --aptos-root data/raw/aptos2019 --prepare-aptos --aptos-output data/manifests/aptos_spec_v1
```

The protocol fixes 60/15/15/10 train/selection/calibration/test group proportions,
seed20261020. Real connected SHA/pHash groups are kept together. Strict mode
rejects conflicting labels; the declared import policy quarantines whole
conflicting components. Actual proportions are reported after rounding. Images and
per-image manifests/duplicate reports remain local. Source labels are preserved.

Standard annotations do not establish patient IDs, eye IDs or quality labels.
Fingerprint components reduce detectable duplicate leakage; they cannot certify
patient-disjoint data. Preserve the new APTOS test partition untouched. Build an
explicitly hashed development-only manifest before invoking the development
trainer, which intentionally rejects a manifest containing held-out test rows.

The historical EfficientNet-B0 candidate in reports/spec_v1 used DeepDRiD. The
new APTOS study trains two genuine frozen-feature candidates with independent
temperature/conformal diagnostics. It still differs from full base-paper
fine-tuning. Before any execution, the runner's default readiness mode verifies
the actual prepared manifests; --execute additionally requires clean committed
source and its declared --source-commit. Completed experiments refuse overwrite.


## Completed parameter-fine-tuning follow-up

The [bounded three-seed pilot](research/APTOS_FINETUNE_V1_RESULTS.md) adapts all
learnable backbone/head parameters with fixed BatchNorm buffers, four epochs per
seed and independent-role temperature/conformal diagnostics. Its actual runner is
scripts/run_aptos_finetune.py; default mode validates readiness without learning.
The committed protocol records exact execution inputs and the pre-execution CPU
budget amendment. Do not overwrite the completed study or its frozen manifests.
Any stronger optimization study needs a new protocol and new output identity.
The active relocated source setting is reused; reserved test remains untouched.
