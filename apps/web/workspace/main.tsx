import { StrictMode, useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import { AlertCircle, CircleCheck, ImagePlus, RefreshCw, ScanEye, ShieldCheck } from 'lucide-react'
import '@fontsource-variable/inter/wght.css'
import '@fontsource-variable/manrope/wght.css'
import mark from '../public/mark.svg?url'
import {
  ACTIONS, GRADES, MAX_UPLOAD_BYTES, MIME_TYPES, PredictionError, analyzeImage, getModelStatus, validateUpload,
  type AnalysisResult, type ModelStatus, type Upload,
} from './inference'
import './workspace.css'

const statusLabels = {
  checking: 'Checking model', ready: 'Model ready', unavailable: 'Model unavailable', offline: 'API unavailable',
}
const progressNotes = {
  checking: 'Checking the local FastAPI service.',
  unavailable: 'Load the trained DR and quality checkpoints and matched calibration, then recheck status.',
  offline: 'Check that the local API is running, then recheck status.',
}
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
const percentage = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 })
const dateTime = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' })

function Results({ result }: { result: AnalysisResult }) {
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
          <p>Suggested action</p>
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

function Workspace() {
  const [status, setStatus] = useState<ModelStatus>({ kind: 'checking' })
  const [attempt, setAttempt] = useState(0)
  const [upload, setUpload] = useState<Upload | null>(null)
  const [preview, setPreview] = useState<{ file: File; url: string } | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    const request = new AbortController()
    let current = true
    const timer = window.setTimeout(() => request.abort(), 5000)
    void getModelStatus(request.signal).then((value) => {
      if (current) setStatus(value)
    }).catch(() => {
      if (current) setStatus({ kind: 'offline' })
    }).finally(() => window.clearTimeout(timer))
    return () => { current = false; request.abort(); window.clearTimeout(timer) }
  }, [attempt])

  useEffect(() => {
    if (!upload) return
    const url = URL.createObjectURL(upload.file)
    setPreview({ file: upload.file, url })
    return () => URL.revokeObjectURL(url)
  }, [upload])

  useEffect(() => () => { active.current?.abort(); active.current = null }, [])

  function resetAnalysis() {
    active.current?.abort()
    active.current = null
    setBusy(false)
    setResult(null)
    setError('')
    setUpload(null)
    setPreview(null)
  }

  function clear() {
    resetAnalysis()
    if (input.current) {
      input.current.value = ''
      input.current.focus()
    }
  }

  function choose(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    resetAnalysis()
    if (!file) return
    const ready = status.kind === 'ready' ? status : null
    try {
      setUpload(validateUpload(file, ready?.types ?? MIME_TYPES, ready?.maxBytes ?? MAX_UPLOAD_BYTES))
    } catch (cause) {
      event.target.value = ''
      setError(cause instanceof Error ? cause.message : 'The selected file could not be read.')
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!upload || status.kind !== 'ready' || active.current) return
    setError('')
    setResult(null)
    try {
      validateUpload(upload.file, status.types, status.maxBytes)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The selected file could not be read.')
      return
    }
    const request = new AbortController()
    active.current = request
    setBusy(true)
    const timer = window.setTimeout(() => request.abort(), 90000)
    try {
      const value = await analyzeImage(upload, request.signal)
      if (active.current === request) setResult(value)
    } catch (cause) {
      if (active.current !== request) return
      if (cause instanceof PredictionError && cause.status === 503) setStatus({ kind: 'unavailable' })
      const message = request.signal.aborted ? 'Analysis timed out. Recheck the service and try again.' :
        'The local service did not return a readable result. Recheck status and try again.'
      setError(cause instanceof PredictionError ? cause.message : message)
    } finally {
      window.clearTimeout(timer)
      if (active.current === request) {
        active.current = null
        setBusy(false)
      }
    }
  }

  function recheck() {
    setStatus({ kind: 'checking' })
    setAttempt((current) => current + 1)
  }

  const previewUrl = upload && preview?.file === upload.file ? preview.url : null
  const maxBytes = status.kind === 'ready' ? status.maxBytes : MAX_UPLOAD_BYTES
  const maxMiB = +(maxBytes / (1024 * 1024)).toFixed(2)
  return (
    <>
      <a href="#main" className="skip-link">Skip to analysis</a>
      <header className="workspace-header">
        <div className="shell header-inner">
          <div className="identity"><img src={mark} alt="" width="36" height="36" />
            <strong>QSART<span>-X</span></strong><span className="workspace-name">Research workspace</span></div>
          <div className="service-status">
            <span className={`status-pill status-${status.kind}`} role="status">{statusLabels[status.kind]}</span>
            <button className="icon-button" type="button" title="Recheck service status" aria-label="Recheck service status"
              onClick={recheck} disabled={status.kind === 'checking' || busy}><RefreshCw size={17} /></button>
          </div>
        </div>
      </header>
      <main className="shell workspace-main" id="main" tabIndex={-1}>
        <h1>Retinal image analysis</h1>
        <p className="page-intro">Run the trained research model on one color fundus photograph.</p>
        <aside className="research-note"><ShieldCheck size={20} aria-hidden="true" />
          <p>Use authorized research images. Predictions and suggested actions are provisional and require clinical judgment.</p></aside>
        {status.kind !== 'ready' && <p className="service-note" role="status">{progressNotes[status.kind]}</p>}
        <div className="analysis-grid">
          <section className="panel image-panel" aria-labelledby="image-heading">
            <h2 id="image-heading">Fundus photograph</h2>
            <form onSubmit={submit}>
              <div className={`preview ${previewUrl ? 'has-image' : ''}`}>
                {previewUrl ? <img src={previewUrl} alt="Selected retinal photograph" /> :
                  <div className="empty-preview"><ImagePlus size={36} aria-hidden="true" />
                    <p>{upload ? 'Preparing preview…' : 'Choose an image to preview it'}</p></div>}
              </div>
              <input ref={input} className="visually-hidden" id="fundus-file" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png"
                aria-describedby="file-hint" onChange={choose} />
              <div className="file-controls">
                <label htmlFor="fundus-file" className="button button-secondary">{upload ? 'Replace image' : 'Choose image'}</label>
                {upload && <button className="button button-text" onClick={clear} type="button">Clear</button>}
              </div>
              <p className="small-copy" id="file-hint">JPG or PNG, up to {maxMiB} MiB.</p>
              {error && <p className="analysis-error" role="alert"><AlertCircle size={17} aria-hidden="true" />{error}</p>}
              <button type="submit" className="button button-primary analyze-button" disabled={!upload || status.kind !== 'ready' || busy}>
                <ScanEye size={19} aria-hidden="true" />{busy ? 'Analyzing image…' : 'Analyze image'}
              </button>
            </form>
            {status.kind === 'ready' && <details className="model-configuration">
              <summary>Loaded models</summary>
              <dl><div><dt>DR</dt><dd>{status.version}</dd></div>
                <div><dt>Quality</dt><dd>{status.qualityVersion}</dd></div>
                <div><dt>Device</dt><dd>{status.device.toUpperCase()}</dd></div></dl>
            </details>}
          </section>
          <section className="panel results-panel" aria-labelledby="results-heading" aria-busy={busy}>
            <div className="panel-heading"><h2 id="results-heading">Analysis results</h2>
              {result && <span className="result-ready" role="status"><CircleCheck size={16} aria-hidden="true" />Complete</span>}</div>
            {result ? <Results result={result} /> : <div className="empty-result" role="status">
              <ScanEye size={37} aria-hidden="true" />
              <p>{busy ? 'The local model is processing the selected image.' : 'Results appear here after analysis.'}</p>
            </div>}
          </section>
        </div>
      </main>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><Workspace /></StrictMode>)
