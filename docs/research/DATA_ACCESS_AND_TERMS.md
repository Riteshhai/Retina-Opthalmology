# Dataset access and permitted use audit

Checked 18 October 2026. This records the terms visible at the named primary sources. It does not replace the terms supplied with a downloaded release or institutional ethics review. Dataset photos, patient identifiers and caches stay outside Git. Code licensing does not automatically relicense images or annotations.

## DeepDRiD

**Suitable for a real-image development pilot now.** The [authors' repository](https://github.com/deepdrdoc/DeepDRiD) publicly serves regular fundus photographs, DR grading annotations and natural image quality annotations. The repository [LICENSE](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/LICENSE) is Creative Commons Attribution ShareAlike 4.0 International. The paper has its own publication license, which must not be confused with the dataset repository license.

Pin `56d8af71319803150d0e7b90533e0918c7bf85ee` rather than the moving `master` branch. The [authors' README](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/README.md) identifies training, validation and evaluation folders, the three study sources and the later release of evaluation labels. No registration or payment gate was encountered for GitHub metadata or the raw-file route.

| Item | Verified route |
| --- | --- |
| Training annotations | [regular-fundus-training.csv](https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/56d8af71319803150d0e7b90533e0918c7bf85ee/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv) |
| Training label definitions | [Readme.docx](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/regular_fundus_images/regular-fundus-training/Readme.docx) |
| Training photograph inventory | [Images directory](https://github.com/deepdrdoc/DeepDRiD/tree/56d8af71319803150d0e7b90533e0918c7bf85ee/regular_fundus_images/regular-fundus-training/Images) |
| Dataset paper | Liu et al., Patterns, 2022, [10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512) |

The regular and ultra-widefield cohorts are distinct imaging regimes. Use regular images first and preserve the released split boundaries. Group all views and both eyes from each verified patient into the same split. Read the supplied label definitions before interpreting numeric quality fields; an eye grade or worst-patient grade must not silently become an image-specific lesion truth. Record original values alongside normalized labels. Archive the license and README, source URL, revision, download date, SHA256 and transformation description. Attribute Liu et al. and the dataset; if redistributing adapted dataset material, carry the license and comply with its attribution and share-alike requirements. This project avoids publishing photos in Git.

## EyeQ and EyePACS

The [authors' EyeQ repository](https://github.com/HzFu/EyeQ) provides quality CSVs for an EyePACS-derived cohort and explains how to obtain and preprocess the original photos. The repository explicitly describes its **code** license as CC BY-NC-SA 4.0; that sentence alone does not establish a separate unrestricted license for all underlying EyePACS photographs or every annotation. The [author paper](https://arxiv.org/abs/1907.05345) confirms the three quality categories and the EyePACS origin.

The [original EyePACS competition route](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) can require an account and acceptance of terms. No credentials were supplied or used and no gate was bypassed. Public labels do not make a third-party photo mirror authorized. Next action: the user obtains photos through the official route under the applicable terms, retains the license/terms snapshot, and places them locally. Join EyeQ annotations by the exact original image identifier. Treat EyeQ and EyePACS as overlapping data sources, including pretraining and any external evaluation. Review the paper's corrected table and the CSV itself instead of hardcoding the README's displayed counts.

## DDR

The [official OIA-DDR repository](https://github.com/nkicsl/DDR-dataset) links [Google Drive](https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC) and Baidu downloads. Its current README explicitly licenses dataset images, annotations and labels as CC BY-NC-SA 4.0 and distinguishes repository code under MIT. This wording was retrieved from the official repository during the audit. Recheck and archive the exact revision when downloading.

The authors describe a ZIP split into ten chunks that must be combined before extraction. The complete archive was not downloaded for this small pilot. The [dataset paper](https://doi.org/10.1016/j.ins.2019.06.011) is Li et al., Information Sciences 501, 511–522, 2019. Preserve gradability/missing-grade semantics. An ungradable class must not be assigned a fabricated ICDR grade. Use DDR as a separate robustness dataset after label definitions, grouping and duplicate checks are complete.

## IDRiD

Primary routes are the [official challenge data page](https://idrid.grand-challenge.org/Data/), [IEEE DataPort record](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [dataset DOI](https://doi.org/10.21227/H25W98) and [Porwal et al. data paper](https://doi.org/10.3390/data3030025). The challenge page returned HTTP 403 in the browsing tool; IEEE DataPort blocked crawler access. The primary download's current click-through terms and archive license were therefore **not independently retrieved** in this audit. Other projects' license descriptions are insufficient to close that check.

Next action: obtain the disease-grading package through the original publisher/challenge interface, save its terms and included license, and verify the 413/103 official train/test assignment before use. Do not confuse the separate lesion segmentation subset with the full grading cohort. IDRiD can support external disease testing after access verification; it does not supply the same natural defect annotations as DeepDRiD. No IDRiD photos were downloaded here.

## What the available data can establish

A small legitimate DeepDRiD subset can establish that ingestion, optimization, serialization, inference and reporting operate on actual fundus photos with real annotations. Its held-out results remain a development pilot. The public cross-sectional datasets do not establish whether a recommended recapture succeeds, whether a specialist subsequently diagnoses correctly, or whether patient outcomes improve. Preserve those as unknown outcomes. A recapture recommendation for severe disease remains a separate possible-delay endpoint.
