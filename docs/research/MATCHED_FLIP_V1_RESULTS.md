# Horizontal-flip inference at matched workloads

Ranking by first-pass entropy gave the largest grade gain on the frozen
DeepDRiD selection cohort. With 42 extra passes, it classified 109 of 168
photos correctly, up from 106. The existing combined ranking classified
108 correctly with the same 42 passes. Flipping all 168 photos corrected
three grades and introduced three new errors, leaving the total at 106.

The study used 168 photographs from 42 development selection patients.
Each comparison assigned two or four extra passes within each of the original
21 requests of eight. It used the saved first-pass and horizontal-flip
logit averages and their existing calibration. The combined ranking exactly
reproduced all 168 saved 25% profile rows apart from timing fields.
[Locked protocol](MATCHED_FLIP_V1_PROTOCOL.md).

## Classification at equal extra-pass counts

| Strategy | Extra passes | Correct grades | Grades corrected / harmed | Correct referability flags | Referability FN / FP | Correct quality labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First pass | 0 | 106 | 0 / 0 | 139 | 13 / 16 | 125 |
| Entropy, 25% | 42 | 109 | 3 / 0 | 141 | 13 / 14 | 125 |
| Referable probability, 25% | 42 | 105 | 0 / 1 | 139 | 13 / 16 | 125 |
| Combined, 25% | 42 | 108 | 2 / 0 | 139 | 13 / 16 | 125 |
| Entropy, 50% | 84 | 108 | 3 / 1 | 143 | 12 / 13 | 126 |
| Referable probability, 50% | 84 | 106 | 1 / 1 | 142 | 12 / 14 | 125 |
| Combined, 50% | 84 | 107 | 2 / 1 | 142 | 12 / 14 | 125 |
| All flips | 168 | 106 | 3 / 3 | 143 | 12 / 13 | 128 |

Referability here uses an image reference grade of at least 2 and a
predicted referable probability of at least .50. At 50%, entropy captured
all four referability corrections seen with all flips. False negatives fell
from 13 to 12 of 74 referable images. False positives fell from 16 to 13 of
94 non-referable images. Every strategy had two false negatives among
21 referable patients and six false positives among 21 non-referable
patients under the any-view patient rule.

At 25%, entropy corrected grades for three patients and introduced no new
grade errors. Increasing to 50% kept those three grade corrections and
introduced a new error. The largest observed grade-3 recall was 24/40,
compared with 23/40 on the first pass.

## Exact uniform selection and attainable changes

Changes count correct images relative to the first-pass profile. Uniform
selection assigns the same number of slots within each request. The full
distribution comes from counting all assignments to the saved predictions.

| Budget | Endpoint | Mean change | Central 95% of random assignments | Attainable range |
| --- | --- | ---: | ---: | ---: |
| 25% | Five-grade | 0 | -2 to +2 | -3 to +3 |
| 25% | Referability | +1 | 0 to +3 | 0 to +4 |
| 50% | Five-grade | 0 | -2 to +2 | -3 to +3 |
| 50% | Referability | +2 | 0 to +4 | 0 to +4 |

At the 25% workload, entropy attained the label-aware upper bound of three
grade corrections and zero new grade errors. Exactly 0.66% of uniform
assignments reached the same net gain. That probability describes slot
assignments on these fixed photos. The possible gains depend on the saved
pairs and the request order.

All eight deterministic profiles retained LAC inclusion of 159/168 images
and 36/42 patients across all views. The ordinal sets contained all five
grades and the deployed ordinal guard recommended specialist review for all
168 images under every profile. The study measures extra-forward counts
and classification changes on reused development patients. The previous
CPU benchmark supplies the separate timing evidence.
[Prior CPU timing report](../../reports/spec_v1/end_to_end_cpu.json).

The independent audit reconstructed all six rankings, both workload anchors
and all four exact random distributions. Fresh CPU inference on the first
eight photographs matched the stored scores, decisions and sets exactly
on both probability paths.
[Aggregate evaluation](../../reports/matched_flip_v1/evaluation.json) ·
[Independent audit](../../reports/matched_flip_v1/independent_audit.json).

## Reproduce

Run both scripts from a clean checkout of
`7df75add5eb18d285abe219abb51f5fadb7f605d` with the bound local score cache,
development manifest, ordinal witness and trained model artifacts.

```text
python scripts/run_matched_flip.py --source-commit 7df75add5eb18d285abe219abb51f5fadb7f605d
python scripts/audit_matched_flip.py
```
