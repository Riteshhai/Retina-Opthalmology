import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { AlertCircle, ImagePlus, ScanEye } from 'lucide-react'
import { MAX_UPLOAD_BYTES, MIME_TYPES, validateUpload, type ModelStatus, type Upload } from './inference'
import { createCase, errorText, sessionEnded, type CaseRecord } from './portal'

export function NewCase({ status, onCreate, onSessionEnd }: {
  status: ModelStatus; onCreate: (record: CaseRecord) => void; onSessionEnd: () => void
}) {
  const [upload, setUpload] = useState<Upload | null>(null)
  const [preview, setPreview] = useState<{ file: File; url: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!upload) return
    const url = URL.createObjectURL(upload.file)
    setPreview({ file: upload.file, url })
    return () => URL.revokeObjectURL(url)
  }, [upload])

  useEffect(() => () => active.current?.abort(), [])

  function clear() {
    setUpload(null)
    setPreview(null)
    setError('')
    if (input.current) {
      input.current.value = ''
      input.current.focus()
    }
  }

  function choose(event: ChangeEvent<HTMLInputElement>) {
    setError('')
    setUpload(null)
    const file = event.target.files?.[0]
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
    try {
      validateUpload(upload.file, status.types, status.maxBytes)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The selected file could not be read.')
      return
    }
    const request = new AbortController()
    active.current = request
    setBusy(true)
    const timeout = AbortSignal.timeout(100_000)
    try {
      const record = await createCase(upload, AbortSignal.any([request.signal, timeout]))
      if (!request.signal.aborted) onCreate(record)
    } catch (cause) {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) return onSessionEnd()
      setError(errorText(cause, 'Check the case queue to see whether the upload completed before trying again.'))
    } finally {
      active.current = null
      if (!request.signal.aborted) setBusy(false)
    }
  }

  const previewUrl = upload && preview?.file === upload.file ? preview.url : null
  const maxBytes = status.kind === 'ready' ? status.maxBytes : MAX_UPLOAD_BYTES
  return (
    <section className="panel upload-panel" aria-labelledby="new-case-title">
      <h2 id="new-case-title">New screening case</h2>
      <p className="small-copy">Analyze and save one authorized color fundus photograph.</p>
      {status.kind !== 'ready' && <p className="service-note" role="status">
        {status.kind === 'checking' ? 'Checking the local model.' : status.kind === 'unavailable'
          ? 'Load the trained DR and quality checkpoints and matched calibration, then recheck status.'
          : 'Check that the local API is running, then recheck status.'}
      </p>}
      <form onSubmit={submit} aria-busy={busy}>
        <div className={`preview ${previewUrl ? 'has-image' : ''}`}>
          {previewUrl ? <img src={previewUrl} alt="Selected retinal photograph" /> :
            <div className="empty-preview"><ImagePlus size={36} aria-hidden="true" />
              <p>{upload ? 'Preparing preview…' : 'Choose an image to preview it'}</p></div>}
        </div>
        <input ref={input} className="visually-hidden" id="fundus-file" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png"
          aria-describedby="file-hint" onChange={choose} disabled={busy} />
        <div className="file-controls">
          <label className="button button-secondary" aria-disabled={busy} htmlFor="fundus-file">{upload ? 'Replace image' : 'Choose image'}</label>
          {upload && <button className="button button-text" onClick={clear} type="button" disabled={busy}>Clear</button>}
        </div>
        <p className="small-copy" id="file-hint">JPG or PNG, up to {+(maxBytes / (1024 * 1024)).toFixed(2)} MiB.</p>
        {error && <p className="analysis-error" role="alert"><AlertCircle size={17} aria-hidden="true" />{error}</p>}
        <button type="submit" className="button button-primary analyze-button" disabled={!upload || status.kind !== 'ready' || busy}>
          <ScanEye size={19} aria-hidden="true" />{busy ? 'Analyzing and saving…' : 'Analyze and save case'}
        </button>
      </form>
      {status.kind === 'ready' && <details className="model-configuration">
        <summary>Loaded models</summary>
        <dl><div><dt>DR</dt><dd>{status.version}</dd></div>
          <div><dt>Quality</dt><dd>{status.qualityVersion}</dd></div>
          <div><dt>Device</dt><dd>{status.device.toUpperCase()}</dd></div></dl>
      </details>}
    </section>
  )
}
