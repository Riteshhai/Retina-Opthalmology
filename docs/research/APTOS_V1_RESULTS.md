# APTOS development v1 — executed 3–4 November 2026

**VERIFIED COMPLETE:** the declared APTOS frozen-feature classifier and independent-role conformal study. EfficientNet-B0 wins the fixed development macro-F1 rule with **78.69% accuracy, macro-F1 0.60334 and QWK 0.84400** on 521 selection images. Grade-3 recall is only **6/26 (23.08%)**. The study provides a genuine APTOS baseline while preserving important grading and grade-conditional coverage failures. Useful automatic triage remains unestablished.

The [protocol](APTOS_V1_PROTOCOL.md) was declared on 3 November before new model outcomes. Both candidates and all uncertainty methods executed from frozen commit `e862700331186e6a754148b33534974d0f716141`. The [complete measured report](../../reports/aptos_v1/evaluation.json) has SHA256 `994a8e78148d6c16d601d61d8774e0d36070b653ebf5549dec093ecbad4924ec`. No classifier, temperature, uncertainty mode or seed from this study was promoted to the application. The selected DeepDRiD DR/quality/calibration/model-card/conformal bytes and production action defaults remain preserved.

## Source integrity and the actual study population

The user confirmed the official Kaggle competition source on 3 November. The 10,215,289,875-byte archive contains 3,662 labeled training PNGs. Only train.csv and those PNGs were extracted into the ignored project data tree. Streaming ZIP CRC checks, actual PNG decoding and all original image-byte hashes passed. Source confirmation and format checks are recorded separately from independent authentication: no Kaggle-published full-archive digest was supplied. The original archive was preserved.

The fixed SHA256-or-central-green-pHash-distance-2 graph has 3,450 original components, 255 candidate pairs and 134 byte-exact pairs. Fifty-one components contain conflicting source grades. Following the predeclared conservative option, **all 148 images in those components were quarantined from every modeling partition**, without voting or relabeling. Strict rejection remains the adapter default. The quarantine includes 35 components of size 2, eight of size 3, three of size 4, three of size 5, one of size 6 and one of size 21.

| Original grade | Source images | Quarantined | Excluded fraction | Retained |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,805 | 42 | 2.33% | 1,763 |
| 1 | 370 | 21 | 5.68% | 349 |
| 2 | 999 | 42 | 4.20% | 957 |
| 3 | 193 | 21 | 10.88% | 172 |
| 4 | 295 | 22 | 7.46% | 273 |

This is a label-dependent exclusion: grade-3 and grade-4 images are removed more often than grade-0 images. Perceptual matches can also be false positives. Results describe the retained **3,514 images / 3,399 fingerprint components**, rather than all original APTOS images or a clinical screening population. Fingerprint components are image-overlap controls, not patient identities. Standard annotations provide no authoritative patient/eye IDs or image-quality targets; no patient counts, patient-disjointness or quality accuracy are reported for APTOS.

| Fixed partition | Images | Fingerprint components | Grade 0 / 1 / 2 / 3 / 4 |
| --- | ---: | ---: | --- |
| Training | 2,115 | 2,048 | 1,059 / 211 / 575 / 105 / 165 |
| Selection | 521 | 507 | 262 / 51 / 142 / 26 / 40 |
| Calibration | 527 | 507 | 265 / 52 / 144 / 25 / 41 |
| Internal reserved test | 351 | 337 | 177 / 35 / 96 / 16 / 27 |

The split uses seed 20261020 and the declared grade-stratified component fractions .60/.15/.15/.10 with fixed integer rounding. Source grades, including eventual reserved-test annotations, were read for integrity, quarantine and stratified assignment. **No reserved-test model inference or outcome evaluation was performed.** The 351-image internal test differs from the inaccessible Kaggle competition labels; competition-test images were not extracted. Neither test population informed classifier fitting, checkpoint choice, temperature fitting or conformal thresholds.

Calibration contains 262 temperature-role images in 253 components and 265 conformal-role images in 254 components. Roles use the declared component-hash ordering. Temperature fitting uses its 262 images; conformal score fitting uses exactly one preselected representative per component, **254 images**. Evaluation retains both all 521 selection images and 507 fixed component representatives. Representative grade supports are **260 / 49 / 135 / 24 / 39**; all-image supports are **262 / 51 / 142 / 26 / 40**. No component crosses these roles or the train/selection/calibration/test boundaries.

## Two classifiers actually trained

Both ImageNet-initialized candidates trained five-way linear heads on 2,115 fitting images, including training-only horizontal-flip feature views, at classifier seed 2026. Backbones and BatchNorm statistics remained fixed. The declared 224px crop/RGB/ImageNet normalization, class-weighted cross entropy, AdamW initial learning rate .01, weight decay .0001, head batch 64, maximum 100 epochs, patience 15, two CPU threads and two workers were retained. Cache-independent head initialization was used. This is a frozen-feature baseline; the base paper’s end-to-end fine-tuning and multi-training-seed reproduction remain unfinished.

| Candidate | Selected / executed epoch | Optimizer updates | Selection accuracy | Macro-F1 | QWK | Unweighted NLL | Checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | 21 / 36 | 2,412 | 78.69% | 0.603341 | 0.844005 | 0.683146 | 16,399,037 |
| MobileNetV3-small | 5 / 20 | 1,340 | 76.97% | 0.570192 | 0.855747 | 0.654518 | 3,881,483 |

The fixed rule selects EfficientNet-B0 by macro-F1, with lower unweighted NLL and declared backbone order as tie breakers. MobileNet has higher QWK and lower raw NLL/ECE in this run; the primary selection rule was not changed to favor those metrics. Epoch and architecture choice used these same selection labels, so the table is development selection evidence. Different APTOS/DeepDRiD populations and protocols do not support a paired improvement claim over earlier experiments. Each architecture has one classifier-training seed; the five later randomization seeds do not estimate training variability.

Selected EfficientNet-B0 class performance, unchanged by positive temperature scaling:

| Grade | Support | Correct grade | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 262 | 252 | 97.30% | 96.18% | 0.967370 |
| 1 | 51 | 30 | 54.55% | 58.82% | 0.566038 |
| 2 | 142 | 100 | 70.92% | 70.42% | 0.706714 |
| 3 | 26 | 6 | 25.00% | 23.08% | 0.240000 |
| 4 | 40 | 22 | 52.38% | 55.00% | 0.536585 |

High overall accuracy and referable-risk discrimination coexist with weak rare-grade classification: grade 3 is correctly classified only six times, and grade 4 only 22/40 times. These failures matter independently of the aggregate accuracy or QWK. No screening action, completed specialist care or clinical safety outcome was measured.

## Independent-role temperature fitting

The selected classifier’s fitted temperature is **1.898052255989193**, learned only from the 262 temperature-role images. Those observations are disjoint from the 254 conformal representatives by observed component. Actual first-pass logits were used consistently for fitting and evaluation; ordinary selection inference matches saved training probabilities with maximum absolute difference **6.22e-07**. No TTA selection or uncertainty-driven classifier choice was introduced.

| Selection measure, 521 images | Raw | Temperature scaled |
| --- | ---: | ---: |
| Unweighted NLL | 0.683146 | 0.580690 |
| ECE, 10 bins | 0.094580 | 0.043193 |
| Multiclass Brier | 0.327335 | 0.302493 |
| Referable AUROC | 0.973704 | 0.976299 |
| Referable AUPRC | 0.948502 | 0.953860 |

Accuracy, macro-F1, QWK and the confusion matrix remain unchanged. These probability improvements do not correct the low grade-3 recall. The separately reported 15-bin temperature-fit ECE is .083295 before and .046207 after; fit NLL is .654727 before and .549482 after on the 262 fitting images. The report’s separate 15-bin selection ECE is .108063 before and .051783 after. These 15-bin numbers use different bins from the primary 10-bin grading table, and fitting diagnostics are not independent test performance.

## All declared conformal outcomes

At each alpha (.10 and .05), the study retained LAC, deterministic adjacent ordinal and randomized ordinal for seeds 2026–2030: **14 comparisons**. Each fits the exact kth threshold on the 254 conformal representatives (rank 230 at alpha .10; rank 243 at .05). Artifacts use the image-unit schema and explicitly identify the one-representative-per-fingerprint-component sampling rule. They do not serialize patient-max scope. Each randomization seed uses separate calibration/evaluation streams; the same draws are reused across alpha and the representative cohort is a subset of the same evaluated draws.

Raw inversion allows empty sets. The common fallback replaces only empty sets with the modal grade; it never removes a label. All fallback rows below have zero empty sets. This can increase observed coverage and mean size, so raw and fallback definitions are kept separate for every method. `Five` counts all-five-grade sets (unchanged by this fallback); `Contiguous` reports the fraction of all observations with nonempty contiguous sets, raw/fallback. Empty sets are not counted as contiguous. Every complete size histogram for sizes 0–5 is retained in the measured JSON.

### All 521 selection images

| Alpha | Method | Random seed | qhat | Raw coverage | Raw mean size | Raw empty | Fallback coverage | Fallback mean size | Five | Contiguous raw / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | LAC | — | 0.753007886 | 86.37% | 1.296 | 0 | 86.37% | 1.296 | 0 | 92.32% / 92.32% |
| 0.10 | Deterministic ordinal | — | 0.998794909 | 93.67% | 3.077 | 22 | 97.89% | 3.119 | 0 | 95.78% / 100.00% |
| 0.10 | Randomized ordinal | 2026 | 0.892131183 | 91.94% | 1.837 | 14 | 94.63% | 1.864 | 5 | 97.31% / 100.00% |
| 0.10 | Randomized ordinal | 2027 | 0.880139383 | 90.79% | 1.745 | 22 | 95.01% | 1.787 | 2 | 95.78% / 100.00% |
| 0.10 | Randomized ordinal | 2028 | 0.893675513 | 91.55% | 1.825 | 17 | 94.82% | 1.858 | 8 | 96.74% / 100.00% |
| 0.10 | Randomized ordinal | 2029 | 0.883860764 | 91.36% | 1.775 | 20 | 95.20% | 1.814 | 4 | 96.16% / 100.00% |
| 0.10 | Randomized ordinal | 2030 | 0.883001757 | 91.17% | 1.772 | 17 | 94.43% | 1.804 | 6 | 96.74% / 100.00% |
| 0.05 | LAC | — | 0.862763679 | 93.47% | 1.614 | 0 | 93.47% | 1.614 | 0 | 85.03% / 85.03% |
| 0.05 | Deterministic ordinal | — | 0.999740066 | 96.93% | 3.491 | 6 | 98.08% | 3.503 | 0 | 98.85% / 100.00% |
| 0.05 | Randomized ordinal | 2026 | 0.958026290 | 96.74% | 2.334 | 4 | 97.50% | 2.342 | 17 | 99.23% / 100.00% |
| 0.05 | Randomized ordinal | 2027 | 0.956464467 | 96.93% | 2.322 | 4 | 97.70% | 2.330 | 17 | 99.23% / 100.00% |
| 0.05 | Randomized ordinal | 2028 | 0.949134752 | 95.97% | 2.232 | 8 | 97.50% | 2.248 | 18 | 98.46% / 100.00% |
| 0.05 | Randomized ordinal | 2029 | 0.942691906 | 96.16% | 2.196 | 7 | 97.50% | 2.209 | 14 | 98.66% / 100.00% |
| 0.05 | Randomized ordinal | 2030 | 0.939268477 | 96.35% | 2.140 | 5 | 97.31% | 2.150 | 11 | 99.04% / 100.00% |

### The 507 fixed selection representatives

| Alpha | Method | Random seed | qhat | Raw coverage | Raw mean size | Raw empty | Fallback coverage | Fallback mean size | Five | Contiguous raw / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | LAC | — | 0.753007886 | 86.59% | 1.290 | 0 | 86.59% | 1.290 | 0 | 92.31% / 92.31% |
| 0.10 | Deterministic ordinal | — | 0.998794909 | 93.49% | 3.063 | 22 | 97.83% | 3.107 | 0 | 95.66% / 100.00% |
| 0.10 | Randomized ordinal | 2026 | 0.892131183 | 91.91% | 1.832 | 14 | 94.67% | 1.860 | 5 | 97.24% / 100.00% |
| 0.10 | Randomized ordinal | 2027 | 0.880139383 | 90.93% | 1.740 | 21 | 95.07% | 1.781 | 2 | 95.86% / 100.00% |
| 0.10 | Randomized ordinal | 2028 | 0.893675513 | 91.52% | 1.817 | 17 | 94.87% | 1.850 | 8 | 96.65% / 100.00% |
| 0.10 | Randomized ordinal | 2029 | 0.883860764 | 91.32% | 1.769 | 20 | 95.27% | 1.809 | 4 | 96.06% / 100.00% |
| 0.10 | Randomized ordinal | 2030 | 0.883001757 | 91.12% | 1.765 | 17 | 94.48% | 1.799 | 6 | 96.65% / 100.00% |
| 0.05 | LAC | — | 0.862763679 | 93.69% | 1.602 | 0 | 93.69% | 1.602 | 0 | 85.01% / 85.01% |
| 0.05 | Deterministic ordinal | — | 0.999740066 | 96.84% | 3.483 | 6 | 98.03% | 3.495 | 0 | 98.82% / 100.00% |
| 0.05 | Randomized ordinal | 2026 | 0.958026290 | 96.84% | 2.323 | 4 | 97.63% | 2.331 | 17 | 99.21% / 100.00% |
| 0.05 | Randomized ordinal | 2027 | 0.956464467 | 97.04% | 2.312 | 4 | 97.83% | 2.320 | 17 | 99.21% / 100.00% |
| 0.05 | Randomized ordinal | 2028 | 0.949134752 | 96.06% | 2.225 | 8 | 97.63% | 2.241 | 18 | 98.42% / 100.00% |
| 0.05 | Randomized ordinal | 2029 | 0.942691906 | 96.25% | 2.187 | 7 | 97.63% | 2.201 | 14 | 98.62% / 100.00% |
| 0.05 | Randomized ordinal | 2030 | 0.939268477 | 96.45% | 2.134 | 5 | 97.44% | 2.144 | 11 | 99.01% / 100.00% |

Across all selection images, the five alpha-.10 randomizations average raw coverage **91.36%**, mean size **1.791** (range 1.745–1.837), and 14–22 raw empties. Fallback averages **94.82%** coverage and mean size **1.825**. At alpha .05, raw averages are **96.43% / 2.245**, and fallback **97.50% / 2.256**. These averages summarize every declared randomization rather than a selected favorable seed. The deterministic predictor is wider and more conservative here; the methods do not have matched empirical coverage, so narrower sets alone do not establish uniform superiority.

### Grade-specific failures remain visible

The following vectors give **numbers of true grades omitted from sets**, ordered grade 0/1/2/3/4, before and after fallback. Divide by the corresponding supports above for empirical failure rates. These are set-inclusion failures, not automatic-clearance or care outcomes; the study performs no triage allocation. All seeds and both cohorts are included.

| Alpha | Method | Seed | All 521: raw -> fallback failures | Representatives 507: raw -> fallback failures |
| --- | --- | ---: | --- | --- |
| 0.10 | LAC | — | 8/15/18/15/15 -> 8/15/18/15/15 | 7/14/18/14/15 -> 7/14/18/14/15 |
| 0.10 | Deterministic ordinal | — | 24/2/0/0/7 -> 2/2/0/0/7 | 24/2/0/0/7 -> 2/2/0/0/7 |
| 0.10 | Randomized ordinal | 2026 | 19/3/5/2/13 -> 5/3/5/2/13 | 18/3/5/2/13 -> 4/3/5/2/13 |
| 0.10 | Randomized ordinal | 2027 | 27/3/5/2/11 -> 5/3/5/2/11 | 25/3/5/2/11 -> 4/3/5/2/11 |
| 0.10 | Randomized ordinal | 2028 | 22/5/5/2/10 -> 5/5/5/2/10 | 21/5/5/2/10 -> 4/5/5/2/10 |
| 0.10 | Randomized ordinal | 2029 | 24/3/5/2/11 -> 4/3/5/2/11 | 23/3/5/2/11 -> 3/3/5/2/11 |
| 0.10 | Randomized ordinal | 2030 | 23/2/8/2/11 -> 6/2/8/2/11 | 22/2/8/2/11 -> 5/2/8/2/11 |
| 0.05 | LAC | — | 6/6/4/9/9 -> 6/6/4/9/9 | 5/6/4/8/9 -> 5/6/4/8/9 |
| 0.05 | Deterministic ordinal | — | 8/2/0/0/6 -> 2/2/0/0/6 | 8/2/0/0/6 -> 2/2/0/0/6 |
| 0.05 | Randomized ordinal | 2026 | 7/2/0/0/8 -> 3/2/0/0/8 | 6/2/0/0/8 -> 2/2/0/0/8 |
| 0.05 | Randomized ordinal | 2027 | 7/2/0/0/7 -> 3/2/0/0/7 | 6/2/0/0/7 -> 2/2/0/0/7 |
| 0.05 | Randomized ordinal | 2028 | 12/3/1/0/5 -> 4/3/1/0/5 | 11/3/1/0/5 -> 3/3/1/0/5 |
| 0.05 | Randomized ordinal | 2029 | 10/2/0/0/8 -> 3/2/0/0/8 | 9/2/0/0/8 -> 2/2/0/0/8 |
| 0.05 | Randomized ordinal | 2030 | 8/2/1/0/8 -> 3/2/1/0/8 | 7/2/1/0/8 -> 2/2/1/0/8 |

Aggregate coverage can conceal severe-grade weakness. At alpha .10, LAC includes the true grade on **86.37%** of all images, below its nominal 90% target, and only **11/26 grade-3 images (42.31%)** and **25/40 grade-4 images (62.50%)**. At alpha .05 its overall coverage is 93.47%, while grade-3 coverage is 17/26 (65.38%) and grade-4 coverage 31/40 (77.50%). No threshold was retuned after these outcomes.

At alpha .10 randomized ordinal grade-4 coverage is only **67.5–75.0%** across seeds on all images, and **66.67–74.36%** on representatives. It remains unchanged by the empty-set fallback even as aggregate coverage rises. At alpha .05, randomized grade-4 coverage is still only **80.0–87.5%** (representatives 79.49–87.18%), alongside roughly 97.5% aggregate fallback coverage. The fallback’s apparent aggregate gain therefore does not resolve these grade-4 exclusions. Deterministic ordinal also excludes 7/40 grade-4 labels at alpha .10 and 6/40 at .05, including after fallback.

Both sampling summaries retain the same qualitative failure: passing or approaching a marginal target does not establish coverage within each disease grade. Unknown patient overlap, grade-stratified partitions, label-dependent quarantine and selection-label reuse prevent blanket iid-image, patient, conditional or external coverage guarantees. No confidence interval or classifier-training variability estimate is inferred from the five randomization draws.

## Provenance, verification and remaining scope

The public report binds this run to the following exact input/artifact bytes. Identifiers, paths, exclusions, per-image logits/sets and trained weights remain in the ignored local artifact trees.

| Binding | SHA256 |
| --- | --- |
| Original user-confirmed archive | `18036845ab76b68d305d6e2dbbaaaf5cd23926be740e1297a8972ac1c6360976` |
| Complete modeling manifest | `1d79240d22237d1934328899d3d6624cd41f6294a15d158a3f62eacf5e7684ed` |
| Development-only manifest | `681799c7ee5f04e988dc607f6cf296fd882c5f55f2603dcd0a2ccd8148951fab` |
| Calibration roles | `65cf2d9206f882265b75b8ce2897bd0555bd7f19d98c3b0a90541b665f2508c1` |
| Source-integrity/split audit | `40dd0fec5c444a283722050ab871073da53d2f016b04f6d274d472de3df6adf8` |
| efficientnet_b0 selected checkpoint | `52574e61db4b68308348aab3794bcc64c55f6d47b0d02932955286c35d190407` |
| mobilenet_v3_small selected checkpoint | `e5e0cddd58ed98e3a0df3cd111aa13a614adfa711cc6c57aeb5bcd1878b37eee` |
| Verified development image-content binding | `1836647751e4d5828648f350e4c21cccde8568ff4c37a68ca8b3c63f2ce3b163` |

Both classifier runs used clean frozen source, legitimate locally preserved official ImageNet weights and source-bound feature caches. Both frozen backbones remained unchanged. Actual selected/last checkpoint hashes, complete histories, config/source/cache/pretrained hashes, all private score digests and original deployment artifact hashes are retained. Legacy checkpoint version strings can collide across datasets; the study identity `aptos-v1-efficientnet_b0-52574e61db4b` and checkpoint/manifest hashes identify this APTOS run.

Recorded feature extraction took 545.10s for EfficientNet-B0 and 513.66s for MobileNet, with respective run/verification durations 574.28s and 549.30s. Classifier training plus calibration/diagnostics took 1435.90s; earlier source import/QA took 856.26s separately. These shared-machine timings include input/cache/verification differences and do not establish architecture speed superiority. The recorded environment is CPU PyTorch 2.14.1, torchvision 0.29.1, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0 and Pillow 10.4.0.

The [independent audit](../../reports/aptos_v1/independent_audit.json) **passed**. It rehashed all 3,662 original images and independently reconstructed the duplicate graph, whole-component exclusions, stable split/representative/role assignments, both candidate histories and winning epochs, saved grading/reliability outcomes, source/artifact bindings and original deployment/master bytes. Every selected/last backbone tensor matches the official pretrained state; direct saved-head/frozen-feature probabilities agree within 1.90e-7. The independent temperature optimum has an NLL gap of 2.66e-11 from the fitted value. All 14 ranks/thresholds and 14,588 individual raw/fallback sets were reconstructed, with maximum numerical threshold error 1.17e-8 and the same actual sets/metrics in both cohorts.

That audit uses saved pHashes for graph reconstruction and saved features/heads for classifier reconstruction. It does not independently re-decode every PNG fingerprint, retrain either model, or execute a new retinal forward pass. These scope limits are retained in the artifact. Final engineering verification is recorded in PROGRESS.md. Derived [candidate comparison](../../reports/aptos_v1/candidate_comparison.png) and [conformal diagnostics](../../reports/aptos_v1/conformal_diagnostics.png) figures accompany the unchanged measured JSON.

This completes the genuine APTOS frozen-feature baseline and its declared uncertainty diagnostic, with unfavorable outcomes preserved. Full fine-tuning, multiple classifier-training seeds, independent reserved-test/external evaluation, authoritative quality supervision and useful validated triage remain separate research requirements. Existing DeepDRiD confirmation remains consumed historical evidence. Formal manuscript drafting remains deferred, and overall QSART scientific acceptance remains **IN PROGRESS**.
