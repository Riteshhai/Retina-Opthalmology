# Literature verification and claim limits

Audit date: 18 October 2026. Reuses all 28 supplied library IDs P01–P28 and adds 15 foundational or close-prior-art records R01–R15. At least 20 supplied titles have independently retrieved primary citation metadata. This is a targeted narrative audit, not an exhaustive systematic review or a replication of every reported number. Metadata verification establishes identity and publication/version status; it does not certify experimental validity.

The full machine-readable audit records exact title, authors where transcribed, year, DOI status, primary URL, verification level and limitations. Publisher blocks and missing DOI fields remain explicit. No numeric performance result from the supplied library is carried into QSART's findings.

## Corrections and unresolved points

- P01: first online in 2023, journal issue in 2024; both dates are recorded.
- P03, P05 and P09: author records state conference/workshop acceptance. Publisher proceedings DOIs were not retrieved; inspected author-version DOIs remain distinct.
- P08: use the publisher's full position-statement title. P28: restore the omitted systematic-review subtitle.
- P02: 2022 preprint and 2023 journal paper are one study. P27: the journal issue is 2023 despite a 2022 DOI/copyright; authors were verified against a coauthor's institutional deposit of the published paper.
- P24 and P25: 2026 records were checked against current primary publisher/arXiv records. They are not assumed authentic merely because the supplied library listed them.
- P10: the linked 2025 Matters Arising commentary was inspected. It questions comparator descriptions, generalization metrics and statistical support; this is a methodological dispute, not a retraction.
- R12: a November 2026 issue date is later than this audit, but a publisher abstract is already indexed. Do not claim the issue has occurred or that full methods were audited.
- R04: official IDRiD dataset download terms remain unverified because the primary interfaces blocked retrieval. A paper's open-access license is not automatically the dataset archive license.

## Individual paper matrix

### P01 Image quality assessment of retinal fundus photographs for diabetic retinopathy in the machine learning era: a review

**2024 · Eye · Peer-reviewed.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10858054/). DOI: `10.1038/s41433-023-02717-3` (verified primary record).

Reviews inconsistent definitions of fundus image quality. Justifies explicit quality-label definitions; review evidence does not prove a new routing policy.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. First online 4 September 2023; Eye volume 38 issue 3 is 2024.

### P02 FundusQ-Net: A regression quality assessment deep learning algorithm for fundus images quality grading

**2023 · Computer Methods and Programs in Biomedicine · Journal publication confirmed in author arXiv record.** [Primary source](https://arxiv.org/abs/2205.01676). DOI: `10.1016/j.cmpb.2023.107522` (verified primary record).

FundusQ-Net estimates continuous fundus image quality with regression. Useful quality baseline; continuous quality is not a measured recapture outcome.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Preprint first submitted in 2022; journal reference is September 2023.

### P03 Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains

**2023 · arXiv:2307.04378 · Author record states accepted MICCAI 2023; publisher DOI not retrieved.** [Primary source](https://arxiv.org/abs/2307.04378). DOI: `10.48550/arXiv.2307.04378` (verified primary record; arXiv DOI denotes author version).

GDRNet addresses visual artifacts, diagnostic variation and imbalance across domains. A grading/generalization comparator; separate held-out domains from training augmentation.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Do not label only an unreviewed preprint; cite inspected author version with acceptance note.

### P04 CauDR: A Causality-inspired Domain Generalization Framework for Fundus-based Diabetic Retinopathy Grading

**2023 · arXiv:2309.15493 · Preprint (verify later venue).** [Primary source](https://arxiv.org/abs/2309.15493). DOI: `10.48550/arXiv.2309.15493` (verified primary record; arXiv DOI denotes author version).

CauDR uses causality-inspired representations for cross-domain DR grading. Claims of causality and generalization remain bounded by the observational benchmark.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P05 Generalizing Across Domains in Diabetic Retinopathy via Variational Autoencoders

**2023 · arXiv:2309.11301 · Author record states accepted MedAGI workshop at MICCAI 2023; publisher DOI not retrieved.** [Primary source](https://arxiv.org/abs/2309.11301). DOI: `10.48550/arXiv.2309.11301` (verified primary record; arXiv DOI denotes author version).

Uses variational autoencoders for domain-invariant DR representations. Supports a simple comparator; does not establish capacity-constrained triage superiority.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P06 A foundation model for generalizable disease detection from retinal images

**2023 · Nature · Peer-reviewed.** [Primary source](https://www.nature.com/articles/s41586-023-06555-x). DOI: `10.1038/s41586-023-06555-x` (verified primary record).

RETFound transfers self-supervised retinal representations to downstream tasks. Optional encoder comparator; foundation pretraining is not required for the CPU pilot.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P07 Real-world evaluation of smartphone-based artificial intelligence to screen for diabetic retinopathy in Dominica: a clinical validation study

**2023 · BMJ Open Ophthalmology · Peer-reviewed.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10749075/). DOI: `10.1136/bmjophth-2023-001491` (verified primary record).

Prospective field evaluation includes the effect of ungradable retinal images. Recapture prompts already exist; retain ungradable cases in operational denominators.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P08 Diabetic retinopathy screening guidelines for Physicians in India: position statement by the Research Society for the Study of Diabetes in India (RSSDI) and the Vitreoretinal Society of India (VRSI)-2023

**2024 · International Journal of Diabetes in Developing Countries · Peer-reviewed position statement.** [Primary source](https://doi.org/10.1007/s13410-023-01296-z). DOI: `10.1007/s13410-023-01296-z` (verified primary record).

Indian professional-society position statement specifies screening pathways. Grade >=2 is a study proxy; DR grade does not by itself capture DME or all referral needs.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P09 Generalizing to Unseen Domains in Diabetic Retinopathy with Disentangled Representations

**2024 · arXiv:2406.06384 (DECO) · Author record states accepted MICCAI 2024; publisher DOI not retrieved.** [Primary source](https://arxiv.org/abs/2406.06384). DOI: `10.48550/arXiv.2406.06384` (verified primary record; arXiv DOI denotes author version).

Disentangles retinal semantics and domain variation for DR grading. Requires external testing; representation gains do not prove triage utility.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P10 RETFound-enhanced community-based fundus disease screening: real-world evidence and decision curve analysis

**2024 · npj Digital Medicine · Peer-reviewed.** [Primary source](https://www.nature.com/articles/s41746-024-01109-5). DOI: `10.1038/s41746-024-01109-5` (verified primary record).

Studies RETFound adaptation and decision curves in community eye screening. Clinical and net-benefit analysis precedes QSART; published commentary questions comparator transparency, generalization metrics and statistical support.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Related primary commentary audited: Chan et al., Matters arising: Utilizing foundation models for developing clinical tools, npj Digital Medicine 8:680 (2025), DOI 10.1038/s41746-025-02065-4, https://www.nature.com/articles/s41746-025-02065-4 . Commentary raises methodological questions; it is not a retraction.

### P11 The performance of a deep learning system in assisting junior ophthalmologists in diagnosing 13 major fundus diseases: a prospective multi-center clinical trial

**2024 · npj Digital Medicine · Peer-reviewed.** [Primary source](https://www.nature.com/articles/s41746-023-00991-9). DOI: `10.1038/s41746-023-00991-9` (verified primary record).

Prospective multicenter trial studies AI assistance for multiple fundus diseases. Human assistance evidence cannot be transferred to an autonomous QSART workflow.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P12 Open ultrawidefield fundus image dataset with disease diagnosis and clinical image quality assessment

**2024 · Scientific Data · Peer-reviewed data descriptor.** [Primary source](https://www.nature.com/articles/s41597-024-04113-2). DOI: `10.1038/s41597-024-04113-2` (verified primary record).

Releases UWF images with diagnoses and multiple quality annotations. UWF is a different modality; disease diagnosis labels do not necessarily provide five-level DR truth.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P13 Smart grading of diabetic retinopathy: an intelligent recommendation-based fine-tuned EfficientNetB0 framework

**2024 · Frontiers in Artificial Intelligence · Peer-reviewed.** [Primary source](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1396160/full). DOI: `10.3389/frai.2024.1396160` (verified primary record).

Fine-tunes EfficientNetB0 and compares transfer-learning classifiers. Architecture and grading are established; a selected Kaggle subset limits transportability.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P14 A systematic review on diabetic retinopathy detection and classification based on deep learning techniques using fundus images

**2024 · PeerJ Computer Science · Peer-reviewed systematic review.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11065411/). DOI: `10.7717/peerj-cs.1947` (verified primary record).

Systematic review summarizes deep learning for fundus DR detection and grading. Contextual review; verify individual studies before reusing their numeric performance.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P15 Novel artificial intelligence for diabetic retinopathy and diabetic macular edema: what is new in 2024?

**2024 · Current Opinion in Ophthalmology · Peer-reviewed review.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11426980/). DOI: `10.1097/ICU.0000000000001084` (verified primary record).

Clinical review discusses emerging AI for DR and diabetic macular edema. Clinical context only; DME cannot be inferred from a five-grade DR output.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P16 Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update

**2024 · Radiology: Artificial Intelligence · Peer-reviewed reporting guideline.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/). DOI: `10.1148/ryai.240300` (verified primary record).

CLAIM 2024 defines reporting requirements for medical imaging AI. Use for provenance, splits, external testing and uncertainty; checklist compliance is not clinical validation.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P17 Diagnostic accuracy of a smartphone-based device (VistaView) for detection of diabetic retinopathy: A prospective study

**2024 · PLOS Digital Health · Peer-reviewed.** [Primary source](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000649). DOI: `10.1371/journal.pdig.0000649` (verified primary record).

Prospective paired-camera study evaluates VistaView DR grading. Camera acquisition and reader agreement affect results; excluded ungradable images affect denominators.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P18 Uncertainty-aware diabetic retinopathy detection using deep learning enhanced by Bayesian approaches

**2025 · Scientific Reports · Peer-reviewed.** [Primary source](https://www.nature.com/articles/s41598-024-84478-x). DOI: `10.1038/s41598-024-84478-x` (verified primary record).

Adds Bayesian uncertainty estimation to DR detection models. Uncertainty is not novelty alone; entropy is not a calibrated probability of error.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P19 Deep learning-based automatic image quality assessment in ultra-widefield fundus photographs

**2025 · BMJ Open · Peer-reviewed.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12096960/). DOI: `10.1136/bmjopen-2025-100058` (verified primary record).

Trains automatic quality assessment for UWF fundus photographs. Modality-specific quality findings should not be generalized automatically to conventional fundus photos.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P20 FundaQ-8: A Clinically-Inspired Scoring Framework for Automated Fundus Image Quality Assessment

**2025 · arXiv:2506.20303 · Preprint.** [Primary source](https://arxiv.org/abs/2506.20303). DOI: `10.48550/arXiv.2506.20303` (verified primary record; arXiv DOI denotes author version).

Proposes a clinically inspired multiparameter quality scale and regression model. Author preprint; framework existence is verified, independent clinical benefit is not established.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P21 Selective Diabetic Retinopathy Screening with Accuracy-Weighted Deep Ensembles and Entropy-Guided Abstention

**2025 · arXiv:2511.05529 · Preprint.** [Primary source](https://arxiv.org/abs/2511.05529). DOI: `10.48550/arXiv.2511.05529` (verified primary record; arXiv DOI denotes author version).

Combines deep ensembles with entropy-guided abstention. Author preprint; high retained-set accuracy must be accompanied by discarded fraction and all-case outcomes.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P22 Diabetic Retinopathy Assessment through Multitask Learning Approach on Heterogeneous Fundus Image Datasets

**2025 · Ophthalmology Science · Journal record verified via primary PMC indexed article.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12167062/). DOI: `10.1016/j.xops.2025.100755` (verified primary record).

Multitask DR analysis includes image quality, laterality and grading. Integrated quality and grading already exist; isolate QSART's decision-policy contribution.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Online March 2025; issue September-October 2025. Browser primary search returned title, DOI and author record.

### P23 The efficacy of artificial intelligence in diabetic retinopathy screening: a systematic review and meta-analysis

**2025 · International Journal of Retina and Vitreous · Peer-reviewed meta-analysis.** [Primary source](https://link.springer.com/article/10.1186/s40942-025-00670-9). DOI: `10.1186/s40942-025-00670-9` (verified primary record).

Meta-analysis pools published AI screening accuracy studies. Pooled performance is heterogeneous and cannot supply QSART's expected accuracy.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P24 Uncertainty-Aware AI-Assisted Diabetic Retinopathy Grading from Fundus Images with Ordinal Conformal Prediction

**2026 · Diagnostics · Journal publication verified from publisher review record.** [Primary source](https://www.mdpi.com/2075-4418/16/16/2645/review_report). DOI: `10.3390/diagnostics16162645` (verified primary record).

Uses ordinal conformal prediction sets for uncertainty-aware DR grading. Marginal coverage assumptions require exchangeability; no external-shift guarantee follows automatically.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Published 19 August 2026. Main HTML retrieval was blocked; publisher review record and version notes verify identity, dates and DOI.

### P25 FOCUS: Benchmarking Retinal Model Generalization from Foundation Vision Encoders to Multimodal LLMs

**2026 · arXiv:2609.33158 (September 2026) · Preprint.** [Primary source](https://arxiv.org/abs/2609.33158). DOI: `10.48550/arXiv.2609.33158` (verified primary record; arXiv DOI denotes author version).

FOCUS benchmarks retinal models across datasets, calibration and quality conditions. September 2026 preprint; generalization and quality robustness cannot be claimed as new alone.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. No numeric QSART finding inferred from this paper.

### P26 Prospective Data Curation Enables High-Performance Artificial Intelligence for Diabetic Retinopathy Screening in a Resource-Limited Setting

**2026 · Ophthalmology Science · Journal publication verified from primary PMC article.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC13218245/). DOI: `10.1016/j.xops.2026.101191` (verified primary record).

Studies prospectively curated UWF data for DR screening model development. The cohort and acquisition process differ from the small retrospective QSART pilot.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Online 15 April 2026; issue June 2026. Full author list remains in the primary record.

### P27 Impact of CLAHE-based image enhancement for diabetic retinopathy classification through deep learning

**2023 · Procedia Computer Science · Conference proceedings article verified from publisher abstract.** [Primary source](https://repository.poliupg.ac.id/431/1/C001_ORIGINAL%20PAPER%20CLAHE.pdf). DOI: `10.1016/j.procs.2022.12.111` (verified primary record).

Compares CLAHE enhancement effects across DR classifiers. Enhancement is not uniformly beneficial; matched preprocessing ablation is needed.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Cite issue year 2023 despite DOI and copyright containing 2022. Author list transcribed from the published paper deposited in coauthor Syamsuddin's institutional repository.

### P28 Diabetic retinopathy screening using machine learning: a systematic review

**2025 · BMC Biomedical Engineering · Peer-reviewed review.** [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12403315/). DOI: `10.1186/s42490-025-00098-0` (verified primary record).

Systematic review includes DR machine-learning methods and model complexity. Review supports resource-aware reporting; it does not demonstrate QSART latency savings.

Audit limit: primary citation metadata and abstract/record audit; detailed experimental audit pending. Supplied title omitted the subtitle; audited title includes 'a systematic review'.

### R01 DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge

**2022 · Patterns · Primary record verified.** [Primary source](https://github.com/deepdrdoc/DeepDRiD). DOI: `10.1016/j.patter.2022.100512` (verified primary record).

Dataset and challenge with DR grading and image-quality assessment. Repeated eyes and views require patient grouping; images do not establish future recapture outcomes.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R02 Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces

**2019 · MICCAI · Primary record verified.** [Primary source](https://arxiv.org/abs/1907.05345). DOI: `10.1007/978-3-030-32239-7_6` (verified primary record).

EyeQ annotates EyePACS quality and evaluates color-space fusion. Quality and grading analysis already exist; EyePACS overlap prevents naive independent testing.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R03 Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening

**2019 · Information Sciences · Primary record verified.** [Primary source](https://github.com/nkicsl/DDR-dataset). DOI: `10.1016/j.ins.2019.06.011` (verified primary record).

DDR dataset and DR grading benchmark. Respect gradability and disease label semantics; dataset and code have distinct licenses.

Audit limit: primary abstract/record audit. Bibliography and dataset terms verified from authors' repository; full paper performance not audited.

### R04 Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research

**2018 · Data · Bibliographic route identified; publisher retrieval rate-limited.** [Primary source](https://doi.org/10.3390/data3030025). DOI: `10.3390/data3030025` (verified primary record).

Indian fundus grading and lesion annotation resource. Download archive license/current click-through terms not retrieved; segmentation subset differs from grading cohort.

Audit limit: primary abstract/record audit. Dataset access and current archive terms remain pending; not included as a completed data acquisition.

### R05 On Calibration of Modern Neural Networks

**2017 · ICML / PMLR 70 · Primary record verified.** [Primary source](https://proceedings.mlr.press/v70/guo17a.html). DOI: `not retrieved` (not retrieved from inspected primary record).

Post-hoc temperature scaling for neural-network calibration. Fit temperature on held-out calibration patients; calibration need not survive dataset shift.

Audit limit: primary abstract/record audit. Publisher provides bibliographic record and pages 1321–1330; no DOI displayed in inspected record.

### R06 Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

**2017 · NeurIPS · Primary record verified.** [Primary source](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). DOI: `not retrieved` (not retrieved from inspected primary record).

Deep ensembles estimate predictive uncertainty. Additional passes cost CPU time; a pilot must measure rather than assume latency benefit.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R07 Selective Classification for Deep Neural Networks

**2017 · NeurIPS / author arXiv version · Primary record verified.** [Primary source](https://arxiv.org/abs/1705.08500). DOI: `10.48550/arXiv.1705.08500` (verified primary record).

Formalizes selective classification with abstention. Coverage and retained risk are established concepts; two operational action costs need separate treatment.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R08 Searching for MobileNetV3

**2019 · ICCV · Primary record verified.** [Primary source](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html). DOI: `not retrieved` (not retrieved from inspected primary record).

Architecture tuned for mobile CPU efficiency. ImageNet/mobile speed does not demonstrate this project's retinal accuracy or host CPU latency.

Audit limit: primary abstract/record audit. CVF record verifies title, authors and venue; DOI withheld because it was not shown in inspected primary record.

### R09 Image Quality-Aware Diagnosis via Meta-Knowledge Co-Embedding

**2023 · CVPR · Primary record verified.** [Primary source](https://openaccess.thecvf.com/content/CVPR2023/html/Che_Image_Quality-Aware_Diagnosis_via_Meta-Knowledge_Co-Embedding_CVPR_2023_paper.html). DOI: `not retrieved` (not retrieved from inspected primary record).

Uses quality annotations in joint diagnostic representation learning. Quality-aware diagnosis predates QSART; the policy problem must establish a narrower contribution.

Audit limit: primary abstract/record audit. CVF record supplies pages 19819–19829; publisher DOI not independently retrieved.

### R10 From discarding to leveraging: quality-aware collaborative learning for robust diabetic retinopathy grading

**2026 · Frontiers in Medicine · Primary record verified.** [Primary source](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1795484/full). DOI: `10.3389/fmed.2026.1795484` (verified primary record).

QGDR studies quality-severity coupling and quality-guided expert routing. Quality-severity interaction and adaptive quality routing are established prior art; no first-of-kind claim for either.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R11 Improving diabetic retinopathy screening using Artificial Intelligence: design, evaluation and before-and-after study of a custom development

**2024 · Author arXiv version; related journal article 2025 · Primary record verified.** [Primary source](https://arxiv.org/abs/2412.14221). DOI: `10.48550/arXiv.2412.14221` (verified primary record).

NaIA-RD integrates grading and quality into an existing clinical screening workflow. Workflow integration and gradability-based output are prior art; a before-after association is not randomized causal evidence.

Audit limit: primary abstract/record audit. Journal article verified at PMC12222278 in 2025; journal DOI not retrieved. BibTeX deliberately cites the verified 2024 author version.

### R12 Multi-Center validation of a Quality-Aware ordinal transformer for robust and Uncertainty-Guided Diabetic retinopathy screening

**2026 · Biomedical Signal Processing and Control · Primary record verified.** [Primary source](https://www.sciencedirect.com/science/article/abs/pii/S1746809426016216). DOI: `10.1016/j.bspc.2026.111067` (verified primary record).

Quality-aware ordinal grading with calibration and uncertainty-guided referral. Directly limits integrated-quality/ordinal/referral novelty; evaluate budgets and independent tuned baselines explicitly.

Audit limit: primary abstract/record audit. Publisher abstract indexed before audit; issue is 1 November 2026, volume 127 article 111067. Full-text methods and publication-online date not independently audited.

### R13 Beyond the Model: Evaluating AI Decision Strategies for Diabetic Retinopathy Screening Under Clinical Constraints

**2026 · Cureus · Primary record verified.** [Primary source](https://assets.cureus.com/uploads/original_article/pdf/489279/20260511-29488-nqzqgj.pdf). DOI: `10.7759/cureus.108630` (verified primary record).

Compares decision strategies under clinical sensitivity constraints. Decision strategy and capacity questions are active prior art; novelty requires a precise same-budget comparison.

Audit limit: primary abstract/record audit. Detailed numeric experimental claims not re-audited.

### R14 Adapting Ordinal-Risk Alignment for Clinically Safe Diabetic Retinopathy Grading

**2026 · MICCAI · Primary record verified.** [Primary source](https://papers.miccai.org/miccai-2026/0030-Paper0916.html). DOI: `not retrieved` (not retrieved from inspected primary record).

RAO-RL applies ordinal learning and asymmetric penalties to DR underestimation. Severity-dependent cost is established; clinical safety in its title does not validate QSART clinically.

Audit limit: primary abstract/record audit. Official accepted-paper record; DOI and final pages marked unavailable/pending on inspected page.

### R15 Towards a Robust and Explainable Pipeline for Diabetic Retinopathy Classification through Quality-Aware GenAI Image Restoration

**2026 · MICCAI OMIA workshop · Primary record verified.** [Primary source](https://papers.miccai.org/miccai-2026-sat/OMIA_001.html). DOI: `not retrieved` (not retrieved from inspected primary record).

Compares restoration and quality-aware triage under synthetic degradation. Synthetic restoration and triage evidence does not establish outcomes of actual clinical recapture.

Audit limit: primary abstract/record audit. Official camera-ready record; DOI and final pages unavailable/pending.


## Selected base-paper follow-up — 21 October 2026

P24 now has a full primary-text and authors-code critical study in
[BASE_PAPER_STUDY.md](BASE_PAPER_STUDY.md). The CSV/JSON record is updated; the
original citation identity/count remain unchanged. Method adaptations and code
discrepancies are explicit, with no exact reproduction claim.
