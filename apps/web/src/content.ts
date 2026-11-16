type Link = { label: string; href: string }

type Section = {
  title: string
  body: string
  links?: Link[]
}

type PageDetails = {
  eyebrow: string
  title: string
  intro: string
  sections: Section[]
}

const repository = 'https://github.com/Riteshhai/Retina-Opthalmology'
const evidenceRevision = '26b5680f93be221bb367fff429a58a23af805e45'
const researchArchive = `${repository}/blob/${evidenceRevision}/docs/research`
const projectIssues = `${repository}/issues`
const study = (label: string, file: string): Link => ({
  label,
  href: `${researchArchive}/${file}.md`,
})

export const screeningActions = [
  {
    code: 'A0',
    name: 'Automated non-referable',
    body: 'A research disposition available only when predefined risk, quality and uncertainty criteria are met.',
  },
  {
    code: 'A1',
    name: 'Request another image',
    body: 'Recommend recapture when the current photograph has a quality issue and the policy permits another image.',
  },
  {
    code: 'A2',
    name: 'Specialist review',
    body: 'Escalate high-risk or unresolved cases. Limited capacity can trigger an explicit overload result.',
  },
]

export const evidence = [
  {
    value: '60.00%',
    label: 'Five-grade accuracy',
    context: '204 of 340 photos in retrospective DeepDRiD internal confirmation.',
  },
  {
    value: '74.85%',
    label: 'External image quality accuracy',
    context: '372 of 497 conventional fundus photos in the MSHF quality study.',
  },
  {
    value: '27.87%',
    label: 'External grade agreement',
    context: '794 of 2,849 FDR26 photos matched the released ALFEDIAM grade code.',
  },
]

export const datasets = [
  {
    name: 'APTOS 2019',
    role: 'Five-grade severity model development',
    status: 'Grading studies recorded. Test split reserved.',
    href: 'https://www.kaggle.com/competitions/aptos2019-blindness-detection/data',
  },
  {
    name: 'DeepDRiD',
    role: 'Disease grading, quality and paired views',
    status: 'Final online split audited and reserved',
    href: 'https://github.com/deepdrdoc/DeepDRiD',
  },
  {
    name: 'IDRiD',
    role: 'External grading and uncertainty assessment',
    status: 'First external evaluation complete',
    href: 'https://idrid.grand-challenge.org/Data/',
  },
  {
    name: 'EyeQ',
    role: 'Optional image quality study',
    status: 'Annotations audited. Source-photo access pending.',
    href: 'https://github.com/HzFu/EyeQ',
  },
]

export const roadmap = [
  {
    title: 'Research foundation',
    body: 'Trained severity and quality models, uncertainty measurement, dual-budget simulations and retrospective studies.',
    status: 'Available',
  },
  {
    title: 'Screening workspace',
    body: 'Secure case uploads, reviewer actions, audit history and structured research reports.',
    status: 'Planned',
  },
  {
    title: 'Adaptive acquisition',
    body: 'Estimate when the benefit of a second retinal view might justify its acquisition cost.',
    status: 'Proposed research',
  },
]

export const references = [
  {
    title: 'DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge',
    venue: 'Patterns',
    year: 2022,
    href: 'https://doi.org/10.1016/j.patter.2022.100512',
  },
  {
    title: 'Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research',
    venue: 'Data',
    year: 2018,
    href: 'https://doi.org/10.3390/data3030025',
  },
  {
    title: 'On Calibration of Modern Neural Networks',
    venue: 'Proceedings of Machine Learning Research',
    year: 2017,
    href: 'https://proceedings.mlr.press/v70/guo17a.html',
  },
  {
    title: 'Uncertainty-Aware AI-Assisted Diabetic Retinopathy Grading from Fundus Images with Ordinal Conformal Prediction',
    venue: 'Diagnostics',
    year: 2026,
    href: 'https://doi.org/10.3390/diagnostics16162645',
  },
]

export const pageDetails: Record<'about' | 'research' | 'publications' | 'contact' | 'responsible-ai', PageDetails> = {
  about: {
    eyebrow: 'About',
    title: 'Image quality matters in retinal screening',
    intro: 'QSART-X is a research platform for studying how image quality, disease severity and uncertainty can guide a screening workflow.',
    sections: [
      {
        title: 'Diabetic retinopathy',
        body: 'Diabetes can damage blood vessels in the retina, the light-sensitive tissue at the back of the eye. Early disease can develop without noticeable symptoms.',
        links: [{
          label: 'National Eye Institute overview',
          href: 'https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/diabetic-retinopathy',
        }],
      },
      {
        title: 'What a fundus photograph can show',
        body: 'A fundus photograph captures part of the back of the eye. Blur, poor illumination, artifacts and limited coverage can obscure features used to grade disease.',
        links: [{ label: 'DeepDRiD data paper', href: references[0].href }],
      },
      {
        title: 'Why routing matters',
        body: 'The project studies whether a policy can prioritize unresolved cases when repeat imaging and specialist review have separate capacity limits.',
      },
    ],
  },
  research: {
    eyebrow: 'Research',
    title: 'Measuring triage under two workload limits',
    intro: 'Can a severity-aware policy reduce missed higher-grade cases at the same imaging and specialist-review workload?',
    sections: [
      {
        title: 'Matched-workload policy comparisons',
        body: 'Compare classifier-only, confidence, entropy, quality and calibrated uncertainty rules with severity-aware routing. Report missed cases, automatic coverage, deferrals and overload at fixed budgets.',
        links: [study('Policy comparison results', 'POLICY_COMPARISON_V1_RESULTS'), study('Patient-level results', 'PATIENT_TRIAGE_V1_RESULTS')],
      },
      {
        title: 'Reliability across cohorts',
        body: 'Separate fitting, calibration and evaluation. Keep known views from the same patient together. Compare grade-specific errors, calibration and uncertainty under degradation and dataset shift.',
        links: [
          study('Controlled degradation', 'ROBUSTNESS_V1_RESULTS'),
          study('Paired external grading', 'FDR26_APTOS_V1_RESULTS'),
          study('Held-out DeepDRiD readiness', 'DEEPDRID_ONLINE_READINESS_V1_RESULTS'),
        ],
      },
      {
        title: 'When is a second view worth acquiring?',
        body: 'The proposed extension compares fixed-view, quality-based, uncertainty-based and value-of-information policies on paired DeepDRiD photos. A simulated acquisition decision sees first-view information only. Paired views are different views of the eye and provide no observed retake outcome.',
      },
    ],
  },
  publications: {
    eyebrow: 'Publications',
    title: 'Research reports and published references',
    intro: 'Read the recorded methods, results and limitations. QSART-X project reports are unpublished research documentation.',
    sections: [
      {
        title: 'Project research reports',
        body: 'The reports document cohort definitions, fixed analysis choices and independent checks. They include adverse findings and unresolved questions.',
        links: [
          study('DeepDRiD internal confirmation', 'CONFIRMATION_V1_RESULTS'),
          study('MSHF external quality evaluation', 'MSHF_QUALITY_V1_RESULTS'),
          study('IDRiD external-domain evaluation', 'IDRID_EXTERNAL_V1_RESULTS'),
          study('Paired FDR26 model comparison', 'FDR26_APTOS_V1_RESULTS'),
        ],
      },
      {
        title: 'Published background',
        body: 'Dataset descriptions and methods for confidence calibration and ordinal conformal prediction.',
        links: references.map(({ title, venue, year, href }) => ({ label: `${title} · ${venue}, ${year}`, href })),
      },
      {
        title: 'Research manuscript',
        body: 'Manuscript preparation is planned after the supporting experiments and reproducibility checks.',
        links: [study('Verified literature matrix', 'LITERATURE_VERIFICATION')],
      },
    ],
  },
  contact: {
    eyebrow: 'Contact and team',
    title: 'Get in touch about QSART-X',
    intro: 'An M.Tech research project in Artificial Intelligence and Machine Learning at VIT Vellore.',
    sections: [
      {
        title: 'Project feedback',
        body: 'Use the project issue tracker for reproducibility questions, documentation corrections or technical feedback. Share aggregate evidence only.',
        links: [{ label: 'Open the project issue tracker', href: projectIssues }],
      },
      {
        title: 'Acknowledgements',
        body: 'The research relies on the published work of dataset authors, graders and challenge organizers. Dataset information and attribution are recorded with the study methods.',
        links: [{ label: 'Read the project documentation', href: `${repository}/tree/${evidenceRevision}` }],
      },
    ],
  },
  'responsible-ai': {
    eyebrow: 'Responsible AI and privacy',
    title: 'Research use and data handling',
    intro: 'QSART-X is a retrospective research and screening-decision-support prototype. Use only authorized research data. Clinical decisions require qualified human review.',
    sections: [
      {
        title: 'Interpret predictions in context',
        body: 'The project uses grade ≥2 as a referability proxy and grade ≥3 as a severity threshold. These research definitions do not cover all clinical referral criteria. The grading model does not assess diabetic macular edema.',
        links: [study('Review the measured limitations', 'FDR26_GRADING_V1_RESULTS')],
      },
      {
        title: 'Separate retrospective and prospective evidence',
        body: 'Image quality labels describe existing photos. Retrospective experiments cannot determine whether a future recapture succeeds or whether specialist review changes patient outcomes.',
        links: [study('Read the patient outcomes protocol', 'PATIENT_TRIAGE_V1_PROTOCOL')],
      },
      {
        title: 'Keep identifiable data private',
        body: 'Public reports contain aggregate results. Keep source images and row-level records in authorized local storage. Do not share images, identifiers or medical records through public forms or issue trackers.',
      },
    ],
  },
}
