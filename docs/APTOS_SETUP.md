# APTOS 2019 acquisition and local preparation

Status on 21 October 2026: **BLOCKED — official labeled export not downloaded**.
The user confirmed it is not available locally. No Kaggle credentials are
configured. The adapter is implemented and tested with generated fixtures; no
APTOS image training, split statistics or experimental results are claimed.

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

After the official files are available, from the QSART project directory run:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_datasets.py --aptos-root data/raw/aptos2019 --prepare-aptos --aptos-output data/manifests/aptos_spec_v1
```

The protocol proposes 60/15/15/10 train/selection/calibration/test proportions,
seed20261020. Real connected SHA/pHash groups are kept together, conflicting labels
block preparation, and actual proportions are reported after rounding. Images and
per-image manifests/duplicate reports remain local. Source labels are preserved.

Standard annotations do not establish patient IDs, eye IDs or quality labels.
Fingerprint components reduce detectable duplicate leakage; they cannot certify
patient-disjoint data. Preserve the new APTOS test partition untouched. Build an
explicitly hashed development-only manifest before invoking the development
trainer, which intentionally rejects a manifest containing held-out test rows.

The completed EfficientNet-B0 candidate in reports/spec_v1 used DeepDRiD. It does
not satisfy the missing APTOS experiment or exact reproduction of the base paper.
