# DeepDRiD regular Set-C data readiness

The authors’ final online challenge split contains 400 regular fundus photos
from 100 patient folders. Each folder contains two photos per eye. Both later
released doctor workbooks join to every photo by original filename stem. The
two views agree on the released grade for all 200 eyes. The patient reference
is derived from the worse eye. [Locked protocol](DEEPDRID_ONLINE_READINESS_V1_PROTOCOL.md) ·
[DeepDRiD data paper](https://doi.org/10.1016/j.patter.2022.100512).

| Grade | Photos | Eyes | Patients by worse-eye grade |
| ---: | ---: | ---: | ---: |
| 0 | 200 | 100 | 43 |
| 1 | 36 | 18 | 9 |
| 2 | 72 | 36 | 20 |
| 3 | 72 | 36 | 18 |
| 4 | 20 | 10 | 10 |

The overall quality workbook includes 220 photos with source flag 0 and 180
with flag 1. It also supplies artifact, clarity and field definition ratings
for every photo. Eye grades may be present when an individual photo received
the lower quality flag. The release records image quality and disease grades.
It provides no observed patient dispositions or recapture outcomes.

## Source verification

The audit matched all 2,000 regular photos across Sets A, B and C to the Git
blob IDs from the authors’ pinned commit. It also authenticated both online
workbooks, the challenge instructions and the repository license. Every photo
decoded as a single-frame RGB JPEG. The 1,184 QSART development rows, 340
confirmation rows and 260 pilot rows each matched authenticated Set-A/B photo
bytes. The reference screen also covered Set-A/B photos outside those
manifests. The online instructions contain a stray reference to training.
The data paper and repository identify this folder as the final online split.
[Pinned release](https://github.com/deepdrdoc/DeepDRiD/tree/56d8af71319803150d0e7b90533e0918c7bf85ee) ·
[CC BY-SA 4.0 license](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/LICENSE).

## Overlap and eligible population

No released patient key occurred in both Set-C and Set-A/B. The pixel screen
found one cross-split candidate between a Set-C photo and a Set-B validation
photo. Their retinal perceptual hashes matched and their file SHA-256 digests
differed. The locked screen excludes the complete Set-C patient group,
leaving 396 photos from 99 patients. The two internal candidates each joined
two views of the same eye and remained eligible. There were no byte-identical
candidate pairs or internal candidates crossing patients.

The eligible group includes 48 patients with a worse-eye grade of 2 or
higher. The fingerprint evidence supports the eligibility decision while
patient identity and undetected overlap remain uncertain. Set-C is a split
from the same release. The audit leaves all 396 eligible photos unscored for
a future patient-level evaluation with frozen models and rules.

The independent audit re-authenticated all 2,000 photographs and four
metadata files, rebuilt both workbooks’ joins and fingerprints, and matched
all three candidate pairs and the patient exclusion.
[Aggregate report](../../reports/deepdrid_online_readiness_v1/evaluation.json) ·
[Independent audit](../../reports/deepdrid_online_readiness_v1/independent_audit.json).

## Reproduce

Use the unmodified author package, the three prior private QSART manifests,
and a Git store containing the pinned author tree. To populate the tree
without downloading the photo blobs, run the following from QSART.

```text
git init --bare artifacts/deepdrid_online_source/author.git
git -C artifacts/deepdrid_online_source/author.git fetch --depth=1 --filter=blob:none --no-tags https://github.com/deepdrdoc/DeepDRiD.git 56d8af71319803150d0e7b90533e0918c7bf85ee
```

Use clean QSART commit `56b41e39641be2e08a9d24b4717c34f5343fa2b8` and
`openpyxl==3.1.5`. Replace `PATH_TO_AUTHOR_PACKAGE` with the source directory
containing `LICENSE` and `regular_fundus_images`.

```text
python -u scripts/prepare_deepdrid_online.py --root PATH_TO_AUTHOR_PACKAGE --source-git artifacts/deepdrid_online_source/author.git --source-commit 56b41e39641be2e08a9d24b4717c34f5343fa2b8
python -u scripts/audit_deepdrid_online.py --root PATH_TO_AUTHOR_PACKAGE --source-git artifacts/deepdrid_online_source/author.git
```
