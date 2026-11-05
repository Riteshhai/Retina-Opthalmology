# EyeQ original-image preparation

The source/annotation audit is metadata-only. It retains the author release and
does not acquire photographs, accept publisher terms or generate model outcomes.
See [protocol](research/EYEQ_DATA_V1_PROTOCOL.md) and the frozen readiness report.

1. Obtain source photographs through the [official 2015 EyePACS competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=test.zip.004)
   under the terms presented to your account. Preserve the source/download and
   accepted-terms record locally. The indexed listing covers88.29GB for the entire
   competition; it exceeds project-volume free space at this audit. Plan storage
   and the required official files before acquisition. The EyeQ subset/unpacked
   footprint has not been measured. Keep the existing project datasets intact.
2. Retain original JPEG filenames and bytes. A flat original root or `train`/`test`
   subfolders is supported. Keep raw data outside Git, for example under
   `C:\PROJECTS FOR MTECH\MAJOR PROJECT CV\eyepacs2015`. The current adapter requires
   original `.jpeg` files; processed PNGs need separate provenance and a declared
   transformation protocol. Quality categories and DR labels remain distinct.
3. Run the private, read-only image inventory from the QSART directory:

   ```powershell
   .\.venv\Scripts\python.exe scripts/audit_eyeq_data.py --image-root "C:\PROJECTS FOR MTECH\MAJOR PROJECT CV\eyepacs2015"
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
