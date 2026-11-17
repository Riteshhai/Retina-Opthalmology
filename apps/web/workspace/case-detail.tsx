import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Download, Printer } from 'lucide-react'
import { ACTIONS, GRADES, isAction } from './inference'
import { dateTime, PredictionResult } from './result'
import { downloadReport, errorText, readImage, saveReview, sessionEnded, type CaseRecord } from './portal'

const gradeLabel = (grade: number) => `${grade} · ${GRADES[grade]}`
const eventLabels: Record<string, string> = {
  case_created: 'Case created',
  review_recorded: 'Assessment recorded',
  action_overridden: 'Action updated',
}

function SavedImage({ id, onSessionEnd }: { id: string; onSessionEnd: () => void }) {
  const [photo, setPhoto] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    const request = new AbortController()
    let url: string | undefined
    void readImage(id, AbortSignal.any([request.signal, AbortSignal.timeout(20_000)])).then((image) => {
      if (request.signal.aborted) return
      url = URL.createObjectURL(image)
      setPhoto(url)
    }).catch((cause: unknown) => {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) onSessionEnd()
      else setError(errorText(cause, 'The saved photograph could not be loaded.'))
    })
    return () => {
      request.abort()
      if (url) URL.revokeObjectURL(url)
    }
  }, [id, onSessionEnd])

  return <section className="panel case-photo" aria-labelledby="saved-image-title">
    <h2 id="saved-image-title">Fundus photograph</h2>
    <div className={`preview ${photo ? 'has-image' : 'empty-preview'}`}>
      {photo ? <img src={photo} alt="Saved fundus photograph for this research case" />
        : <p role={error ? 'alert' : 'status'}>{error || 'Loading saved photograph…'}</p>}
    </div>
  </section>
}

function ReviewForm({ record, onReviewed, onSessionEnd }: {
  record: CaseRecord; onReviewed: (record: CaseRecord) => void; onSessionEnd: () => void
}) {
  const [grade, setGrade] = useState('')
  const [action, setAction] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (active.current || grade === '') return
    const proposed = isAction(action) ? action : null
    if (proposed !== null && !notes.trim()) {
      setError('Enter the reason for the action override in the review notes.')
      return
    }
    const request = new AbortController()
    active.current = request
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const updated = await saveReview(record.id, { grade: Number(grade), action: proposed, notes: notes.trim() },
        AbortSignal.any([request.signal, AbortSignal.timeout(20_000)]))
      if (request.signal.aborted) return
      onReviewed(updated)
      setGrade('')
      setAction('')
      setNotes('')
      setMessage('Assessment saved.')
    } catch (cause) {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) onSessionEnd()
      else setError(errorText(cause, 'Reopen the case to see whether the assessment was saved before trying again.'))
    } finally {
      active.current = null
      if (!request.signal.aborted) setBusy(false)
    }
  }

  return <form className="review-form" onSubmit={submit} aria-busy={busy}>
    <h3>Record an assessment</h3>
    <p className="small-copy">Enter your research assessment and use case identifiers only in the notes.</p>
    <div className="review-fields">
      <label>Assessed grade<select value={grade} onChange={(event) => setGrade(event.target.value)} required disabled={busy}>
        <option value="">Select a grade</option>
        {GRADES.map((_, index) => <option key={index} value={index}>{gradeLabel(index)}</option>)}
      </select></label>
      <label>Override action<select value={action} onChange={(event) => setAction(event.target.value)} disabled={busy}>
        <option value="">Keep current action</option>
        {Object.entries(ACTIONS).map(([code, label]) => <option key={code} value={code}>{label}</option>)}
      </select></label>
    </div>
    <label>Review notes {action && <span className="small-copy">(required for an override)</span>}
      <textarea value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={1500} rows={4}
        required={action !== ''} disabled={busy} />
    </label>
    {error && <p className="analysis-error" role="alert">{error}</p>}
    <div className="review-submit"><button className="button button-primary" disabled={busy || grade === ''}>
      {busy ? 'Saving assessment…' : 'Save assessment'}</button>
      {message && <p className="small-copy" role="status">{message}</p>}
    </div>
  </form>
}

export function CaseDetail({ record, canReview, canExport, onUpdate, onSessionEnd }: {
  record: CaseRecord; canReview: boolean; canExport: boolean;
  onUpdate: (record: CaseRecord) => void; onSessionEnd: () => void
}) {
  const [exportMessage, setExportMessage] = useState('')
  const [exporting, setExporting] = useState(false)
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  async function exportJson() {
    if (active.current) return
    const request = new AbortController()
    active.current = request
    setExporting(true)
    setExportMessage('')
    try {
      const blob = await downloadReport(record.id, AbortSignal.any([request.signal, AbortSignal.timeout(20_000)]))
      if (request.signal.aborted) return
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `qsart-case-${record.id}.json`
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 5000)
    } catch (cause) {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) onSessionEnd()
      else setExportMessage(errorText(cause, 'The private case report could not be downloaded.'))
    } finally {
      active.current = null
      if (!request.signal.aborted) setExporting(false)
    }
  }

  return <article className="case-detail" aria-labelledby="case-id-title">
    <div className="case-title">
      <div><h2 id="case-id-title">Case details</h2><p className="case-code">{record.id}</p>
        <time className="small-copy" dateTime={record.createdAt}>{dateTime.format(new Date(record.createdAt))}</time>
      </div>
      {canExport && <div className="report-actions">
        <button className="button button-secondary" onClick={exportJson} type="button" disabled={exporting}>
          <Download size={16} aria-hidden="true" />{exporting ? 'Preparing report…' : 'Download JSON report'}</button>
        <button className="button button-secondary" onClick={() => window.print()} type="button">
          <Printer size={16} aria-hidden="true" />Print / Save as PDF</button>
      </div>}
    </div>
    <p className="print-only">QSART-X screening research report. Predictions and suggested actions are provisional.
      Reviewer assessments are user-entered research observations.</p>
    {exportMessage && <p className="analysis-error" role="alert">{exportMessage}</p>}
    <div className="analysis-grid">
      <SavedImage id={record.id} onSessionEnd={onSessionEnd} />
      <section className="panel" aria-labelledby="saved-results-title">
        <h2 id="saved-results-title">Model results</h2>
        <PredictionResult result={record.prediction} />
      </section>
    </div>
    <section className="panel review-panel" aria-labelledby="review-title">
      <div className="review-heading">
        <div><h2 id="review-title">Reviewer assessments</h2>
          <p className="small-copy">User-entered research assessments.</p></div>
        <dl className="current-action"><dt>Current workflow action</dt><dd>{ACTIONS[record.effectiveAction]}</dd></dl>
      </div>
      {record.reviews.length > 0 ? <ol className="review-history">{record.reviews.map((review) =>
        <li key={review.id}>
          <div className="review-meta"><strong>{review.reviewer}</strong>
            <time dateTime={review.createdAt}>{dateTime.format(new Date(review.createdAt))}</time></div>
          <p>Grade {gradeLabel(review.grade)} <span className="review-badge">
            {review.gradeDisagreement ? 'Grade differs from model' : 'Grade agrees with model'}</span></p>
          {review.action && <p>{review.isOverride ? 'Action override' : 'Action recorded'} · {ACTIONS[review.action]}</p>}
          {review.notes && <p className="review-notes">{review.notes}</p>}
        </li>)}</ol> : <p className="small-copy no-review">No assessment recorded.</p>}
      {canReview && <ReviewForm record={record} onReviewed={onUpdate} onSessionEnd={onSessionEnd} />}
    </section>
    <section className="panel audit-panel" aria-labelledby="audit-title">
      <h2 id="audit-title">Action history</h2>
      <ol className="audit-list">{record.audit.map((entry, index) => <li key={`${entry.createdAt}-${index}`}>
        <strong>{eventLabels[entry.event] ?? entry.event.replaceAll('_', ' ').toLowerCase()}</strong>
        <span>{entry.actor}</span>
        <time dateTime={entry.createdAt}>{dateTime.format(new Date(entry.createdAt))}</time>
      </li>)}</ol>
    </section>
  </article>
}
