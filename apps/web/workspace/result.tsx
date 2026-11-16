import { ACTIONS, GRADES, type AnalysisResult } from './inference'

const reasonLabels: Record<string, string> = {
  PREDICTED_REFERABLE_GRADE: 'The predicted grade meets the study’s referral definition.',
  HIGH_REFERABLE_PROBABILITY: 'The grade 2 or higher estimate meets the policy review threshold.',
  HIGH_SEVERE_PROBABILITY: 'The grade 3 or higher estimate meets the policy review threshold.',
  EMPTY_CONFORMAL_SET: 'The conformal method returned an empty grade set.',
  CONFORMAL_SET_INCLUDES_REFERABLE: 'The conformal set includes a grade that requires review under the research policy.',
  HIGH_PREDICTIVE_UNCERTAINTY: 'Predictive uncertainty meets the policy review threshold.',
  INSUFFICIENT_IMAGE_QUALITY: 'The model estimated insufficient image quality for provisional clearance.',
  PROVISIONAL_NONREFERABLE: 'The result meets the policy criteria for provisional non-referable disposition.',
}
export const percentage = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 })
export const dateTime = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' })

export function PredictionResult({ result }: { result: AnalysisResult }) {
  const reasons = result.reasons.flatMap((reason) => reasonLabels[reason] ? [reasonLabels[reason]] : [])
  const measurements = [
    { label: 'Referable DR estimate', detail: 'Grade 2 or higher', value: percentage.format(result.referable) },
    { label: 'Severe DR estimate', detail: 'Grade 3 or higher', value: percentage.format(result.severe) },
    { label: 'Poor image quality', detail: 'Model-estimated probability', value: percentage.format(result.poorQuality) },
    { label: 'Predictive uncertainty', detail: 'Normalized entropy from 0 to 1', value: result.entropy.toFixed(3) },
  ]
  return (
    <div className="result-body">
      <div className="case-outcome">
        <div>
          <p>Predicted grade</p>
          <h3>Grade {result.predictedGrade} · {GRADES[result.predictedGrade]}</h3>
          <span>{percentage.format(result.confidence)} top-grade probability</span>
        </div>
        <div className="result-action">
          <p>Model suggestion</p>
          <strong>{ACTIONS[result.action]}</strong>
        </div>
      </div>
      {reasons.length > 0 && <ul className="action-reasons">{reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
      <dl className="metrics-grid">
        {measurements.map(({ label, detail, value }) => (
          <div className="metric" key={label}><dt>{label}</dt><dd>{value}</dd><small>{detail}</small></div>
        ))}
      </dl>
      <section aria-labelledby="grade-probabilities">
        <h3 id="grade-probabilities">Five-grade probabilities</h3>
        <ul className="grade-list">
          {result.gradeProbabilities.map((probability, grade) => (
            <li className={grade === result.predictedGrade ? 'predicted' : ''} key={grade}>
              <span>{grade} · {GRADES[grade]}</span>
              <progress max={1} value={probability} aria-label={`Grade ${grade} probability`} />
              <span>{percentage.format(probability)}</span>
            </li>
          ))}
        </ul>
      </section>
      <section className="result-details" aria-labelledby="analysis-details">
        <h3 id="analysis-details">Analysis details</h3>
        <dl>
          <div><dt>Ordinal conformal set</dt><dd>{result.conformalGrades === null ? 'Unavailable' :
            result.conformalGrades.length ? result.conformalGrades.join(', ') : 'Empty set'}</dd></div>
          <div><dt>Model version</dt><dd>{result.modelVersion}</dd></div>
          <div><dt>Policy version</dt><dd>{result.policyVersion}</dd></div>
          <div><dt>Inference time</dt><dd><time dateTime={result.timestamp}>{dateTime.format(new Date(result.timestamp))}</time></dd></div>
          <div><dt>Model processing</dt><dd>{result.processingTime.toFixed(1)} ms</dd></div>
        </dl>
        <p className="small-copy">This recommendation applies to one image. Review and recapture capacities apply to cohort decisions.</p>
      </section>
    </div>
  )
}
