# FDR26 v3 external grading data audit

The verified release contains 3,000 original RGB JPEGs and 3,000 grading
rows. An exact filename join matched 2,999. The unmatched photo is a
byte-identical copy of a different, labeled photo and carries a duplicate
suffix. The unmatched workbook entry carries grade 0. The audit preserved
both records and excluded the two-photo fingerprint component.

| ALFEDIAM grade | Workbook | Exact image joins |
| --- | ---: | ---: |
| 0 | 505 | 504 |
| 1 | 701 | 701 |
| 2 | 583 | 583 |
| 3 | 632 | 632 |
| 4 | 579 | 579 |

The overlap screen identified 189 internal candidate pairs, including 161
byte-identical pairs. Of those exact pairs, 62 had different released
grades on the two joined rows. In total, 71 fingerprint components contained
conflicting grades.
Three perceptual matches involved two FDR26 photos and APTOS references.
All three had pHash distance two. The 5,186 reference rows also included
the frozen DeepDRiD development and confirmation manifests.

| Population | Eligible photos | One per fingerprint component |
| --- | ---: | ---: |
| Primary exclusions | 2,849 | 2,742 |
| Exclusions extended through filename prefixes | 2,761 | 2,673 |

The workbook yielded 1,918 eight-character filename prefixes, of which 855
had multiple photos. Every group had contiguous two-digit numbering and
consistent demographic fields. The count matches the patient total in the
data article. The prefix-to-patient mapping remains unconfirmed. The
second population extends exclusions through these prefixes to test the
effect of possible grouping. [Data article](https://doi.org/10.1016/j.dib.2026.113379).

This is a data readiness audit. A later grading study would need to specify
how it compares model grade codes with the released ALFEDIAM stages.
The workbook supplies neither an adjudicated quality endpoint nor a
diabetic macular edema endpoint. [Locked protocol](FDR26_READINESS_V1_PROTOCOL.md).

## Reproduce

Install the workbook reader with `python -m pip install -e ".[fdr26]"`.
Save the Zenodo record JSON as `data/raw/fdr26_v3/record.json`. Put
`Fundus_FDR26.zip` and `metadata_classification_DR.xlsx` in the same folder.
The runner uses the four local reference manifests in
`configs/idrid_data_v1.json`. [Zenodo record API](https://zenodo.org/api/records/23101200).

```text
python scripts/prepare_fdr26.py --source-commit 887f9ec82d1e0744060da7f3ae4afa812826bc07
```

Run `python scripts/audit_fdr26_readiness.py` from audit commit
`4e36d8e0a6ea65294d41f59ef4dc8e7f4b79ab64` or a later checkout with the
same audit source. The auditor canonicalizes undirected internal pair
endpoints and binds its own execution revision separately.

[Aggregate evaluation](../../reports/fdr26_readiness_v1/evaluation.json) ·
[Independent audit](../../reports/fdr26_readiness_v1/independent_audit.json).

FDR26 v3 is attributed to Sanae Lamti, Noureddine Kerzazi, Rachid Sekhsoukh
and Asmae Guechati. [Author deposit](https://doi.org/10.5281/zenodo.23101200) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
