# MSHF v2 external overall-quality results

The frozen DeepDRiD quality head correctly classified 372 of 497 eligible
MSHF photographs (74.85% accuracy), compared with 58.15% for always predicting
good. Poor-quality AUROC was .8973. At the fixed poor-probability threshold
above .50, the model flagged 181 of 208 poor photographs and 98 of 289 good
photographs. Errors varied sharply by source. No model, temperature or
threshold was fitted on MSHF.

The [protocol](MSHF_QUALITY_V1_PROTOCOL.md) and executable audit were locked
at `92b93077534eff891bbdec7c02e7afee611fb941` before predictions. The selected
MobileNetV3-Small head used its 224-pixel preprocessing and the first-pass
DeepDRiD temperature of 1.0349426972, with a single CPU forward per eligible
photograph.

## Annotation and overlap audit

The archive contained 500 decodable photographs across the four conventional
source groups. The 422 DR-XJU and DR-ZJU images decode as JPEG; the 78 Glaucoma
and Healthy images decode as TIFF despite the `.jpg` suffix. All were single-frame
RGB. The study decodes originals offline with the frozen head’s preprocessing.
All 500 matched the individual workbook exactly by filename and
had three binary overall ratings. The paper’s majority rule produced 209 poor
and 291 good labels; 448 ratings were unanimous.

| Source | Images | Poor / good by majority | Poor / good in paper Table 4 | Unanimous | Cached summary disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| DR-XJU | 235 | 117 / 118 | 117 / 118 | 204 | 16 |
| DR-ZJU | 187 | 41 / 146 | 40 / 147 | 168 | 146 |
| Glaucoma | 52 | 51 / 1 | 50 / 2 | 50 | 1 |
| Healthy | 26 | 0 / 26 | 0 / 26 | 26 | 0 |
| Total | 500 | 209 / 291 | 207 / 293 | 448 | 163 |

The two extensionless names in the DR-XJU summary were matched by unique
stems for this audit. The cached DR-ZJU sheet coded all 187 joined rows poor;
its overall formulas refer to three empty cells per row. The individual
ratings are complete and implement the stated majority rule. The protocol
fixed their use before seeing any model outcomes.

Two internal perceptual pairs produced 498 components. The ten reference
pairs involved three MSHF photos and all matched APTOS reference
fingerprints. Every candidate had pHash distance two and none was byte
identical. The protocol conservatively excluded those three photos (two
DR-ZJU good, one Glaucoma poor). No internal component had conflicting
majority labels. The main population had 497 photos (208 poor, 289 good),
the representative sensitivity population 495 (208 poor, 287 good), and
the unanimous sensitivity population 445 (188 poor, 257 good).

## Frozen threshold and discrimination

| Population or main source group | Poor / good | Poor flagged | Good passed | Accuracy | Always-good accuracy | Poor AUROC | Poor AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 208 / 289 | 181/208 | 191/289 | 74.85% | 58.15% | 0.8973 | 0.8818 |
| One per component | 208 / 287 | 181/208 | 189/287 | 74.75% | 57.98% | 0.8966 | 0.8818 |
| Unanimous ratings | 188 / 257 | 170/188 | 181/257 | 78.88% | 57.75% | 0.9347 | 0.9288 |
| DR-XJU | 117 / 118 | 114/117 | 31/118 | 61.70% | 50.21% | 0.8685 | 0.8703 |
| DR-ZJU | 41 / 144 | 21/41 | 134/144 | 83.78% | 77.84% | 0.9153 | 0.7640 |
| Glaucoma | 50 / 1 | 46/50 | 1/1 | 92.16% | 1.96% | 0.9600 | 0.9992 |
| Healthy | 0 / 26 | — | 25/26 | 96.15% | 100.00% | — | — |

Pooled sensitivity and specificity were 87.02% and 66.09%, giving 76.55%
balanced accuracy. DR-XJU accounted for 87 of the 98 false flags. DR-ZJU
accounted for 20 of the 27 missed poor photographs, despite strong within-group
ranking (AUROC .9153). Glaucoma specificity rests on a single good photograph,
and sensitivity and ranking cannot be estimated for Healthy. The one-per-component
view changed pooled results little. The unanimous view performed better on its
different subset and does not establish a performance improvement on the full
cohort. All raw and calibrated confusion matrices and subgroup metrics are in the
[aggregate report](../../reports/mshf_quality_v1/evaluation.json).

## Probability calibration

| Population | NLL raw → calibrated | Brier raw → calibrated | ECE raw → calibrated |
| --- | ---: | ---: | ---: |
| Main | 0.4556 → 0.4565 | 0.3042 → 0.3040 | 0.0711 → 0.0742 |
| One per component | 0.4565 → 0.4574 | 0.3051 → 0.3048 | 0.0717 → 0.0749 |
| Unanimous ratings | 0.3956 → 0.3984 | 0.2551 → 0.2559 | 0.0762 → 0.0798 |

The frozen temperature changed no class decisions or discrimination metrics.
On the main cohort it slightly reduced Brier score while increasing log loss
and confidence ECE. All three calibration metrics worsened in the unanimous
subset. These results do not support assuming that DeepDRiD calibration will
transfer to another source.

Sources differ in acquisition, encoding and label mix; pooled performance can
reflect that mix. The workbook has no patient linkage, and the fingerprint
screen cannot establish patient independence. These are image-level estimates
for overall quality. The study evaluated neither DR grading nor triage utility
and did not change the deployed policy.

The independent audit reconstructed 500 archived image and annotation rows,
the candidate screen against 5,186 reference fingerprints and all 30 metric
cells. It checked the 497 stored probability pairs against the saved logits
and reproduced logits exactly on eight fresh forwards (two per source).
[Independent audit receipt](../../reports/mshf_quality_v1/independent_audit.json).

## Reproduce the locked run

Install the local research dependency with `python -m pip install -e ".[mshf]"`.
Download file 39485878 from the [Figshare v2 deposit](https://doi.org/10.6084/m9.figshare.21507564.v2)
to `data/raw/mshf_v2/MSHF.zip`. Check the release size and both digests in the
[protocol](MSHF_QUALITY_V1_PROTOCOL.md). Save the [official version metadata](https://api.figshare.com/v2/articles/21507564/versions/2)
alongside the archive as `figshare_v2.json`. The local `receipt.json` records
`article_id` (21507564), `version` (2), `file_id` (39485878), `doi`
(`10.6084/m9.figshare.21507564.v2`), `license` (`CC BY 4.0`), `license_url`
(`https://creativecommons.org/licenses/by/4.0/`), `download_url`
(`https://ndownloader.figshare.com/files/39485878`), `bytes` (1136891149),
`publisher_md5` and `archive_sha256` from the verified archive, and
`metadata_sha256` from the saved metadata. The runner checks the archive and
the release fields independently of that receipt.

The four frozen reference manifests
and the selected DeepDRiD quality checkpoint/calibration are local inputs
named by the runner. In a clean checkout of the bound source, run:

```text
python scripts/evaluate_mshf_quality.py --source-commit 92b93077534eff891bbdec7c02e7afee611fb941
python scripts/audit_mshf_quality.py
```

Source attribution: Kai Jin, Zhiyuan Gao, Xiaoyu Jiang, Yaqi Wang, Xiaoyu Ma,
Yunxiang Li and Juan Ye. The [MSHF v2 data](https://doi.org/10.6084/m9.figshare.21507564.v2)
are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [data paper](https://doi.org/10.1038/s41597-023-02188-x) describes the
author majority rule. The archive, per-image ratings, overlap witnesses and
predictions remain local.
