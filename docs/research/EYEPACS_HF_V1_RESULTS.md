# User-selected EyePACS cache results — 6 November 2026

**VERIFIED COMPLETE for acquisition and annotation compatibility.** EyeQ model
validation and the overall scientific project remain IN PROGRESS. Source is the
user-selected [bumbledeep/eyepacs](https://huggingface.co/datasets/bumbledeep/eyepacs),
pinned to revision `3e94599d7e41d9992a8362cc0467eb397e11210f`.
Acquisition began from `25fb7969ef93cb23fede7d8f547cb4343e1a05a3`.
The recovery implementation and subsequent checks use
`c1e52fd21f43e38de78007e1376055276be5865d`. The
[declared acquisition plan](EYEPACS_HF_V1_PLAN.md) remains preserved.

All fourteen Parquet files are cached locally: **6,535,643,877 bytes, 35,108 unique
original image names**. Every file matches its pinned size and SHA256; the cached
dataset card and local receipt also match. Initial long responses disconnected;
the revised downloader resumed the actual source-bound prefixes using 32-MiB
HTTP ranges and bounded retries. Earlier partials, successful files and failure
logs were preserved. No alternative dataset or remote executable code was used.

The mirror retains disease labels. EyeQ's original three quality labels were
joined separately by exact original filename, without assigning labels to other
rows or altering author partitions. Both source partitions remain external-only.
The complete compatibility result is:

| EyeQ author partition | Annotated | Matched | Missing | Missing Good / Usable / Reject |
| --- | ---: | ---: | ---: | --- |
| Train | 12,543 | 12,529 | 14 | 0 / 0 / 14 |
| Test | 16,249 | 0 | 16,249 | 8,471 / 4,558 / 3,220 |
| Combined | 28,792 | 12,529 | 16,263 | 8,471 / 4,558 / 3,234 |

The matched rows contain 8,347 Good, 1,876 Usable and 2,306 Reject annotations.
All shared DR grades agree. Another 22,579 mirror rows have no EyeQ quality
annotation. **This cache cannot supply the full EyeQ test set.** The fourteen
missing train images all carry Reject labels; the mirror card describes removal
of nearly black/unmaskable images, but the cause of each missing row was not
individually established. Do not infer unbiased quality-class retention.

The independent auditor hashes the entire local cache again, reconstructs all
35,108 name/DR records and all 28,792 raw author CSV rows, verifies the private
join and compares every identity/DR record to the earlier bounded remote metadata
scan. All 307 preserved historical artifacts, original models and master copies
remain fixed. Initial auditor setup was corrected for the author CSV's unnamed
index and a local-path guard that incorrectly matched HTTPS; these changes did
not alter acquisition, annotations, source code or neural results. Failed local
attempt logs remain available privately.

The remote scan fetched 1,291,755 bytes while selecting name/DR columns. Parquet
footer read-ahead may also transfer adjacent compressed payload bytes; this is
not a claim that zero image-payload bytes traversed the network. No JPEG was
decoded during either full metadata reconstruction or the production label join.
A separate preselected three-payload format/geometry check is reported separately
from that metadata scope. It performs no inference or quality reassessment and
does not establish complete image integrity or original pixel equivalence.

The card describes cropping, resizing up to 1024 pixels and removed images.
Its MIT claim is recorded separately from underlying original-photograph terms,
which this community card does not independently establish. Original image-byte
equivalence, transported quality truth, patient/pixel independence and evaluation/
training readiness remain false. No mirror model predictions, training, quality
validation or external test outcomes were produced. Original-image acquisition,
overlap checks and a locked quality evaluation protocol remain later work.

The downloader is explicit and reusable:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_eyepacs_hf.py
.\.venv\Scripts\python.exe scripts/fetch_eyepacs_hf.py --download --workers 4
.\.venv\Scripts\python.exe scripts/fetch_eyepacs_hf.py --audit-cache
```

The first command previews the source and space plan. Downloading resumes only
matching partials and reuses verified completed files. Run one acquisition
process per cache; its internal workers handle separate shards. The default is
two workers. Cache payloads, source filenames, annotation joins and extracted
samples remain under ignored data/artifact directories. Application startup
does not fetch datasets. This behavior follows the
[Hugging Face dataset download model](https://huggingface.co/docs/hub/datasets-downloading)
with additional exact-manifest/range/integrity checks.

The final genuine-model/conformal suite passed **618 tests in 130.82 seconds**,
with no failures/errors/skips and one upstream AnyIO deprecation warning; 25
targeted checks passed in 11.29 seconds. Ruff, Compose and whitespace checks pass.
Both native and rebuilt Docker services passed actual JPEG, batch, second-pass,
conformal, simulator and overload checks. The current image matches all 70 runtime
source/configuration files and includes pinned PyArrow 25.0.1. The primary Docker
app is running on workstation ports 18000/18501; the temporary verification stack
was removed after the primary deployment passed. No new interactive browser
rendering claim is made. Production models and the 20-record research registry
remain unchanged.

Evidence: [acquisition and split coverage](../../reports/eyeq_hf_v1/acquisition.json),
[independent cache/join audit](../../reports/eyeq_hf_v1/independent_audit.json),
[bounded payload/offline reuse checks](../../reports/eyeq_hf_v1/payload_and_reuse.json),
[primary container/source proof](../../reports/eyeq_hf_v1/primary_runtime.json),
[actual deployment checks](../../reports/eyeq_hf_v1/primary_deployment.json) and
[verification](../../reports/eyeq_hf_v1/verification.json).
The separate [224/384 feasibility result](APTOS_384_FEASIBILITY_RESULTS.md)
does not create a new trained candidate or a paper-reproduction claim.
