# DeepDRiD regular Set-C data readiness

Audit the authors’ regular-fundus online challenge release from DeepDRiD. The
data paper describes Set-C as the final online evaluation with 400 photos
from 100 patients. The repository later published the doctor annotations.
This is a held-out split of the same dataset release. The supplied challenge
instructions contain an inconsistent reference to training. The study uses
the split designation in the data paper and the authors’ repository.
[Data paper](https://doi.org/10.1016/j.patter.2022.100512) ·
[Authors’ pinned repository](https://github.com/deepdrdoc/DeepDRiD/tree/56d8af71319803150d0e7b90533e0918c7bf85ee) ·
[CC BY-SA 4.0 license](https://github.com/deepdrdoc/DeepDRiD/blob/56d8af71319803150d0e7b90533e0918c7bf85ee/LICENSE).

Verify the local online photographs, both label workbooks, challenge README,
license, and all 1,600 regular Set-A/B photographs against their blob IDs in
the authors’ pinned Git tree. Hash the original bytes with SHA-256 and decode
single-frame RGB JPEGs. Verify that images used by QSART’s development,
confirmation and pilot manifests agree with the authenticated source by
identifier and SHA-256. Bind the exact manifests. Screen the complete source
training and validation photo sets, including rows excluded from earlier work.

Join the grade and quality workbooks by exact original filename stem. Ignore
submission templates. Verify the published label domains and retain both views
per eye. Require that each online patient folder has exactly left 1, left 2,
right 1 and right 2, with agreeing DR grades across the two views of each eye.
The patient grade is derived as the worse eye grade. The source’s overall
quality flag is a photo assessment. The study preserves the defect scores
and assigns no missing clinical or image-level disease labels.

Screen online photos for byte equality or central-green retinal pHash distance
at most two, both within Set-C and against all authentic Set-A/B photos.
Exclude the entire online patient when the source patient key occurs in
Set-A/B, any of their photos has a Set-A/B counterpart candidate, or an
internal pair links two patient groups. Exclude both online groups in the last
case. Exclude an online patient if left and right views share identical file
bytes. Retain other same-patient pairs and report their counts separately.
The fingerprint screen finds candidates and cannot establish patient identity
or rule out overlap. All eligible patients retain all four views.

Independently reconstruct source authentication, file and label inventory,
both pair sets, and all patient eligibility counts. Publish aggregate counts
and provenance. Keep identifiers, photo-level labels and fingerprints in
ignored private storage. This audit leaves Set-C unscored. Any later evaluation
must lock the model and patient-level rules before running inference.
