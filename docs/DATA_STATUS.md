# Data pipeline evidence — 2026-10-18

**VERIFIED COMPLETE — scoped real-image engineering pilot and ingestion software.**

Official source metadata, annotation definitions and license were downloaded from
DeepDRiD commit `56d8af71319803150d0e7b90533e0918c7bf85ee`. Primary-source
repository browsing and direct inspection of the original Word table established
the quality code direction and eye-level grade meaning. The original strict
metadata parser failed on 11 actual training rows with side/view conflicts;
the new parser quarantines these explicitly and reuses the starter normalizer
for verified rows. All three existing parser tests remain passing.

Local discovery checked the workspace, Downloads top-level candidate names, and
one level of the plausible `archive`, `archive (1)`, and `files` directories. No
existing retinal-image dataset was found in that bounded search. This does not
claim that every drive/folder was searched. No large recursive drive scan ran.

The actual deterministic pilot download acquired **260 source JPEGs, 65 supplied
patient IDs, 145,693,718 bytes (138.9 MiB)**. It includes 50 official-training
patients and 15 distinct official-validation patients, sampled before model
training, with all four images each. Image dimensions range across five decoded
shapes, from width 1734–2232 and height 1725–1984. Full annotation metadata was
audited: 1,200 training rows / 300 patients, 400 validation rows / 100 patients,
with no cross-source patient overlap. Only the pilot's actual images were acquired.

| Experiment partition | Source | Images | Patients | Poor / good quality |
|---|---|---:|---:|---:|
| train | official training | 160 | 40 | 104 / 56 |
| calibration | official training | 20 | 5 | 8 / 12 |
| validation for selection | official training | 20 | 5 | 14 / 6 |
| locked holdout | official validation | 60 | 15 | 30 / 30 |

All partitions are patient disjoint; all normalized grades and three defect
labels are present for the pilot. The pilot excludes the four subjects affected
by ambiguous source rows. Its class-enriched sampling cannot represent real
screening prevalence or substantiate research-performance claims.

`python scripts/validate_data.py data/processed/deepdrid/pilot_manifest.csv --out data/processed/deepdrid/pilot_manifest.verified_audit.json`
completed with **exit 0**, `ready_for_training=true`, no missing/corrupt images,
no quarantined rows, no byte-exact duplicate groups, and no cross-partition
perceptual duplicate candidates at central-green DCT pHash distance <=2. Four
within-patient/view near pairs remain logged within their respective partitions.
Whole-image dHash initially flagged 623 cross-partition pairs, dominated by dark
round fundus borders; crop-aware DCT fingerprinting resolved these outline-driven
collisions. These checks reduce duplicate leakage risk, and do not prove its
absence or establish clinical identity.

The pilot CSV was frozen before model fitting and deliberately preserved during
the improved perceptual audit. Its SHA256 is
`e7e349903964f3c64935092b205647a2153b79c6ae18e4d0a4179cf54af285f0`.
The separate verified audit records that exact hash. The older adjacent
`pilot_manifest.audit.json` retains the historical dHash warnings; use
`pilot_manifest.verified_audit.json` for final readiness evidence. The selection
JSON SHA256 is `d34424ece4bf898690482ebeacce590a5c38d552027062b95c8fdd29a51f46cb`.
The provenance JSON records pinned source URLs and every JPEG SHA256/size.

Executable commands implemented:

```powershell
python scripts/prepare_data.py --download-pilot --out data/processed/deepdrid/pilot_manifest.csv
python scripts/prepare_data.py --config configs/data.yaml --root PATH_TO_DATA --train-csv TRAIN.csv --validation-csv VALIDATION.csv --out NEW_MANIFEST.csv
python scripts/validate_data.py data/processed/deepdrid/pilot_manifest.csv --out data/processed/deepdrid/pilot_manifest.verified_audit.json
.venv/Scripts/python.exe -m pytest tests/test_deepdrid.py tests/test_data_pipeline.py -q
```

Final targeted test execution: **16 passed in 6.70 seconds**, using project
`.venv` with pandas 3.0.5, NumPy 2.4.6, Pillow 10.4.0 and PyYAML 6.0.3. Coverage
includes valid decoding, corruption/missing images, missing fields, unknown
quality codes, conflicting annotation quarantine, source overlap, split leakage,
SHA/perceptual duplicates, deterministic splits, unsafe paths and shared PIL
crop/pad/resize. `git check-ignore` confirms raw images and processed manifests
are excluded. No image or patient manifest was committed.

The model agent received the manifest path, frozen hash, quality direction,
verified defect mappings and preprocessing interface before fitting. Shared
`qsart.preprocessing.preprocess_image(PIL_image, image_size=...)` returns a PIL
RGB square; source pixels remain unchanged. Optional defect heads require code
mapping because the ordinal codes are not contiguous class IDs.

**PARTIALLY COMPLETE — research dataset integration.** Full-size DeepDRiD image
acquisition, external IDRiD/DDR experiments, EyeQ/EyePACS access and registration,
external-domain overlap checks and human adjudication of the source annotation
conflicts are outstanding. The pilot is sufficient for executing actual training
and integration; it is insufficient for final scientific conclusions. The data
agent did not run or tune on holdout predictions. Models, clinical validity and
deployment are separate acceptance gates tracked by their owners.
