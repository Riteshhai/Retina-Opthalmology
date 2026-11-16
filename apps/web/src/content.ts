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
const evidenceRevision = 'da3b01a38bacdd49e06e1fc59bbdf9d361604ec3'
const researchArchive = `${repository}/blob/${evidenceRevision}/docs/research`
const projectIssues = `${repository}/issues`
const study = (label: string, file: string): Link => ({
  label,
  href: `${researchArchive}/${file}.md`,
})
const setCResults = study('DeepDRiD Set-C validation', 'SET_C_VALIDATION_V1_RESULTS')
const pairedViewResults = study('DeepDRiD paired-view acquisition', 'PAIRED_VIEW_V1_RESULTS')

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
    value: '99 of 99',
    label: 'Set-C patients recommended for review',
    context: 'At full image review capacity. The deployed policy overloaded at the three lower tested caps.',
  },
  {
    value: '8 → 6',
    label: 'Missed referable eyes',
    context: 'First view alone versus value ranking with eight extra fields on 84 development eyes. All matched cost intervals included zero.',
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
    status: 'Set-C evaluated on 99 eligible patients. Paired-view analysis used Set-A development data.',
    href: 'https://github.com/deepdrdoc/DeepDRiD',
  },
  {
    name: 'IDRiD',
    role: 'External grading and uncertainty assessment',
    status: 'External evaluation completed. Outcomes already inspected.',
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
    status: 'Available locally',
  },
  {
    title: 'Adaptive acquisition',
    body: 'Compare when to acquire a second recorded retinal field at fixed workloads in the research lab.',
    status: 'Retrospective study audited',
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
    title: 'Review capacity and second-view acquisition',
    intro: 'Two retrospective studies measure the workload of a frozen screening policy and the value of an additional recorded retinal field.',
    sections: [
      {
        title: 'Review capacity on held-out DeepDRiD Set-C',
        body: 'The frozen ordinal policy required review for all 396 photographs from 99 patients. It overloaded at 30%, 60%, and 90% image review caps. At full capacity the policy recommended specialist review for all 99 patients. All 12 policy comparisons with matched workloads at both image and patient level made identical decisions.',
        links: [setCResults, study('Set-C protocol', 'SET_C_VALIDATION_V1_PROTOCOL')],
      },
      {
        title: 'Acquiring a second recorded field',
        body: 'On 84 eyes from 42 DeepDRiD development patients, the first-view value ranking missed six referable eyes at eight extra fields, compared with eight using the first view alone. All 16 matched-workload cost intervals included zero. The study replayed existing fields and measured no new captures or retake outcomes.',
        links: [pairedViewResults, study('Paired-view protocol', 'PAIRED_VIEW_V1_PROTOCOL')],
      },
      {
        title: 'Reliability across cohorts',
        body: 'Separate studies measured grade-specific errors, calibration and uncertainty under controlled degradation and dataset shift. Each report gives the population and reference labels used.',
        links: [
          study('Earlier policy comparisons', 'POLICY_COMPARISON_V1_RESULTS'),
          study('Earlier patient-level triage', 'PATIENT_TRIAGE_V1_RESULTS'),
          study('Controlled degradation', 'ROBUSTNESS_V1_RESULTS'),
          study('Paired external grading', 'FDR26_APTOS_V1_RESULTS'),
          study('DeepDRiD Set-C eligibility audit', 'DEEPDRID_ONLINE_READINESS_V1_RESULTS'),
        ],
      },
    ],
  },
  publications: {
    eyebrow: 'Publications',
    title: 'Research reports and published references',
    intro: 'Read the recorded methods, results and limitations. QSART-X project reports are unpublished research documentation.',
    sections: [
      {
        title: 'Manuscript for author review',
        body: 'The unpublished draft covers review-capacity limits on held-out DeepDRiD Set-C and recorded second-field acquisition on development patients.',
        links: [
          study('Read the manuscript draft', 'QSART_X_MANUSCRIPT'),
          study('Trace results to the evidence table', 'QSART_X_EVIDENCE_TABLE'),
        ],
      },
      {
        title: 'Project research reports',
        body: 'The reports document cohort definitions, fixed analysis choices and independent checks. They include adverse findings and unresolved questions.',
        links: [
          setCResults,
          pairedViewResults,
          study('DeepDRiD internal confirmation', 'CONFIRMATION_V1_RESULTS'),
          study('MSHF external quality evaluation', 'MSHF_QUALITY_V1_RESULTS'),
          study('IDRiD external-domain evaluation', 'IDRID_EXTERNAL_V1_RESULTS'),
          study('Paired FDR26 model comparison', 'FDR26_APTOS_V1_RESULTS'),
        ],
      },
      {
        title: 'Published background',
        body: 'Dataset descriptions and methods for confidence calibration and ordinal conformal prediction.',
        links: [
          ...references.map(({ title, venue, year, href }) => ({ label: `${title} · ${venue}, ${year}`, href })),
          study('Verified literature matrix', 'LITERATURE_VERIFICATION'),
        ],
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
