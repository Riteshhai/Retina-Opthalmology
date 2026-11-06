# EyeQ original-image preparation

The source/annotation audit is metadata-only. It retains the author release and
does not acquire photographs, accept publisher terms or generate model outcomes.
See [results](research/EYEQ_DATA_V1_RESULTS.md), the preserved
[protocol](research/EYEQ_DATA_V1_PROTOCOL.md) and the frozen readiness report.

1. Obtain source photographs through the [official 2015 EyePACS competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=test.zip.004)
   under the terms presented to your account. Preserve the source/download and
   accepted-terms record locally. The indexed listing covers88.29GB for the entire
   competition; it exceeds project-volume free space at this audit. Plan storage
   and the required official files before acquisition. The EyeQ subset/unpacked
   footprint has not been measured. Keep the existing project datasets intact.
2. Retain original JPEG filenames and bytes. A flat original root or `train`/`test`
   subfolders is supported. Keep raw data outside Git, for example under
   `..\eyepacs2015` relative to the QSART directory. The current adapter requires
   original `.jpeg` files; processed PNGs need separate provenance and a declared
   transformation protocol. Quality categories and DR labels remain distinct.
3. Run the private, read-only image inventory from the QSART directory:

   ```powershell
   .\.venv\Scripts\python.exe scripts/audit_eyeq_data.py --image-root "..\eyepacs2015"
   ```

   Results stay in ignored `artifacts/eyeq_data_v1/private_image_inventory`.
   An existing output is preserved; use a new `--inventory-output artifacts/...`
   directory for a later inventory. Outputs cannot be written into original-image
   roots. Exact source names, decode, dimensions, bytes and retinal fingerprints
   are checked; originals are unchanged.
4. Before model evaluation, complete source/terms evidence and cross-source image
   overlap checks against the previously used DeepDRiD/APTOS/IDRiD populations.
   Lock the external cohort, model states, preprocessing and explicit quality-label
   harmonization before predictions. Both author splits are external-only in this
   adapter; existing training loaders reject this manifest.

The author CSVs can be verified without source images:

```powershell
.\.venv\Scripts\python.exe scripts/audit_eyeq_data.py --fetch-metadata
```

That command fetches only seven immutable public author metadata/code files.
It never downloads images, imports/runs author code, retrieves author model results
or creates a new training target. `--execute` is a one-time frozen metadata audit;
completed measured reports are preserved.

## User-selected Hugging Face cache

The selected [bumbledeep/eyepacs mirror](https://huggingface.co/datasets/bumbledeep/eyepacs)
is approximately 6.54 GB in fourteen pinned Parquet files. It contains cropped/
resized EyePACS images and DR grades. Original filenames are present, allowing an
exact-name compatibility audit with our separate EyeQ quality CSVs. Its quality
labels after transformation and any missing images must be reviewed before use.

From the project directory, preview the source/space plan:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_eyepacs_hf.py
```

Download once, verify SHA256 and audit the metadata join:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_eyepacs_hf.py --download
```

Rerunning resumes interrupted transfers and reuses matching cached files.
Transfers use 32-MiB ranges with bounded retries after interrupted connections.
`--workers 4` can increase concurrency; the default is two files at a time.
The cache is data/external/eyepacs_hf_v1; receipts, private annotation joins and
audit summaries remain under ignored data/artifacts. App startup does not fetch
the dataset. `--audit-cache` repeats the read-only metadata audit without network
acquisition. [Acquisition plan](research/EYEPACS_HF_V1_PLAN.md).

Keep original-source and transformed-mirror evidence distinct. The mirror card's
license claim does not independently settle underlying image terms. The cache
alone does not make external quality evaluation or training ready. Original-image
preparation above remains applicable when the official photographs are available.
