# FDR26 v3 external grading data readiness

Audit the authors' FDR26 v3 fundus photographs and grading workbook from
Zenodo record 23101200 under CC BY 4.0. The data article describes 3,000
photographs from 1,918 patients at Mohammed VI University Hospital in Oujda,
Morocco. It reports consensus grading by four ophthalmologists.
[Author deposit](https://doi.org/10.5281/zenodo.23101200) ·
[Data article](https://doi.org/10.1016/j.dib.2026.113379).

Verify the record, file sizes and publisher MD5 checksums for
`Fundus_FDR26.zip` and `metadata_classification_DR.xlsx`. Bind their local
SHA256 digests. Join the workbook's image identifier to original
`images/<id>.jpg` members by exact stem. The ZIP directory preflight found
2,999 exact joins, one unmatched workbook row and one unmatched archive
member with a duplicate-style suffix. Preserve the unmatched records and
assign no grade to the orphan. Preserve the five
source ALFEDIAM stages and verify that numeric and textual labels agree.
Read the original JPEG bytes without extracting them. Verify single-frame
RGB decoding and record dimensions and fingerprints. This audit uses the
classification workbook and photographs. Its scope excludes vessel masks.

Use SHA256 equality or central-green retinal pHash distance at most two
to identify candidates within FDR26 and against the four frozen manifests
from the IDRiD readiness audit. Those references contain the 3,662 APTOS
annotated images, including quarantined images, and 1,524 DeepDRiD images
from development and confirmation. Compute transitive internal components.
Flag a whole component if it contains conflicting joined source grades,
a reference match candidate or the unlabelled archive member. Preserve every
original row and label. Count
eligible images and one representative per remaining component, selected
by lexicographic image identifier. The screen finds candidates and can
miss related images. Components provide no evidence of patient identity.

The workbook contains no explicit patient or laterality key, quality label
or DME field. In a metadata-only check, the eight-character image prefixes
matched the published patient total. The last two characters numbered each
prefix's images from 01, and workbook demographics were consistent within
prefixes. Count these patterns as evidence of a possible grouping. Verify
the patterns independently and treat the prefixes as unconfirmed.
In a sensitivity view, extend each primary exclusion through shared prefixes
and fingerprint components until no additional images are linked. Retain
the fingerprint representatives for the remaining images. Keep this view
separate from the primary overlap exclusions.
The source grade distribution reflects this dataset's composition. ALFEDIAM
and ICDR have differing lesion criteria. A future external diagnostic must
name the released ALFEDIAM labels and declare any grade-code comparison
before scoring.
[Classification comparison](https://www.has-sante.fr/jcms/c_1028304/en/argumentaire-depistage-de-la-retinopathie-diabetique-par-lecture-differee-de-photographies-du-fond-d-oeil).

Run an independent reconstruction of the labels, original image inventory,
fingerprints and exclusions. Publish aggregate counts and input bindings.
Keep image identifiers, per-image grades and fingerprints in ignored local
storage. The audit does not execute model inference or assess clinical triage.
