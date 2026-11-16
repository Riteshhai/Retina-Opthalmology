import { StrictMode, useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowLeft, ArrowRight, ClipboardList, LockKeyhole, Plus, RefreshCw, ShieldCheck } from 'lucide-react'
import '@fontsource-variable/inter/wght.css'
import '@fontsource-variable/manrope/wght.css'
import mark from '../public/mark.svg?url'
import { CaseDetail } from './case-detail'
import { ACTIONS, GRADES, getModelStatus, type ModelStatus } from './inference'
import {
  ROLES, errorText, getSession, listCases, readCase, sessionEnded, signIn, signOut,
  type Account, type CaseRecord, type CaseSummary,
} from './portal'
import { dateTime, percentage } from './result'
import { NewCase } from './upload'
import './workspace.css'

function Header({ account, onSignOut }: { account?: Account; onSignOut?: () => void }) {
  return <>
    <a href="#main" className="skip-link">Skip to workspace</a>
    <header className="workspace-header">
      <div className="shell header-inner">
        <div className="identity"><img src={mark} alt="" width="36" height="36" />
          <strong>QSART<span>-X</span></strong><span className="workspace-name">Research workspace</span></div>
        {account && <div className="account">
          <div><strong>{account.username}</strong><span>{ROLES[account.role]}</span></div>
          <button className="button button-secondary" type="button" onClick={onSignOut}>Sign out</button>
        </div>}
      </div>
    </header>
  </>
}

function SignIn({ notice, onSignIn }: { notice: string; onSignIn: (account: Account) => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (active.current) return
    const form = event.currentTarget
    const fields = new FormData(form)
    const request = new AbortController()
    active.current = request
    setError('')
    setBusy(true)
    try {
      const account = await signIn(String(fields.get('username')).trim(), String(fields.get('password')),
        AbortSignal.any([request.signal, AbortSignal.timeout(12_000)]))
      if (!request.signal.aborted) onSignIn(account)
    } catch (cause) {
      if (!request.signal.aborted) setError(errorText(cause, 'Sign-in could not reach the local portal.'))
    } finally {
      const field = form.elements.namedItem('password') as HTMLInputElement | null
      if (field) field.value = ''
      active.current = null
      if (!request.signal.aborted) setBusy(false)
    }
  }

  return <main id="main" tabIndex={-1} className="shell workspace-main sign-in-main">
    <section className="panel sign-in-card" aria-labelledby="sign-in-title">
      <LockKeyhole className="sign-in-icon" size={23} aria-hidden="true" />
      <h1 id="sign-in-title">Sign in to QSART-X</h1>
      <p className="page-intro">Use your local research account to access screening and review.</p>
      {notice && <p className="small-copy" role="status">{notice}</p>}
      <form onSubmit={submit} aria-busy={busy}>
        <label>Username<input name="username" autoComplete="username" required maxLength={96} /></label>
        <label>Password<input type="password" name="password" autoComplete="current-password" required maxLength={512} /></label>
        {error && <p className="analysis-error" role="alert">{error}</p>}
        <button className="button button-primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </form>
    </section>
  </main>
}

const filters = [
  'All cases', 'Pending specialist review', 'Pending recapture', 'Assessments recorded',
] as const

function CaseQueue({ onOpen, onSessionEnd }: { onOpen: (id: string) => void; onSessionEnd: () => void }) {
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  const [filter, setFilter] = useState(0)
  const [attempt, setAttempt] = useState(0)
  const [error, setError] = useState('')
  useEffect(() => {
    const request = new AbortController()
    void listCases(AbortSignal.any([request.signal, AbortSignal.timeout(12_000)])).then((rows) => {
      if (!request.signal.aborted) setCases(rows)
    }).catch((cause: unknown) => {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) onSessionEnd()
      else setError(errorText(cause, 'The case queue could not be loaded.'))
    })
    return () => request.abort()
  }, [attempt, onSessionEnd])

  const specialist = cases?.filter((row) => row.effectiveAction === 'SPECIALIST_REVIEW' && !row.reviewed)
  const recapture = cases?.filter((row) => row.effectiveAction === 'RECAPTURE' && !row.reviewed)
  const reviewed = cases?.filter((row) => row.reviewed)
  const rows = [cases, specialist, recapture, reviewed][filter] ?? []

  function recheck() {
    setError('')
    setCases(null)
    setAttempt((value) => value + 1)
  }

  return <section className="panel queue-panel" aria-labelledby="queue-title">
    <div className="queue-heading">
      <div><h2 id="queue-title">Case queue</h2>
        <p className="small-copy">Pending cases appear first by action priority and estimated severe DR risk.</p></div>
      <button className="icon-button" onClick={recheck} type="button" aria-label="Refresh case queue" title="Refresh case queue">
        <RefreshCw size={16} aria-hidden="true" /></button>
    </div>
    {error && <p className="analysis-error" role="alert">{error}</p>}
    {!error && cases === null && <p className="queue-empty" role="status">Loading cases…</p>}
    {cases !== null && <>
      <div className="queue-filters" role="group" aria-label="Filter cases">
        {filters.map((label, index) => <button type="button" className="queue-filter" key={label}
          aria-pressed={filter === index} onClick={() => setFilter(index)}>
          <strong>{[cases, specialist, recapture, reviewed][index]?.length}</strong><span>{label}</span>
        </button>)}
      </div>
      {rows.length === 0 ? <p className="queue-empty" role="status">{filter ? 'No cases match this filter.' : 'No screening cases yet.'}</p>
        : <ul className="queue-list">{rows.map((row) => <li key={row.id}>
          <button className="queue-row" type="button" onClick={() => onOpen(row.id)}>
            <span><strong>Case {row.id.slice(0, 8)}</strong><small><time dateTime={row.createdAt}>{dateTime.format(new Date(row.createdAt))}</time></small></span>
            <span><strong>{ACTIONS[row.effectiveAction]}</strong><small>{row.reviewed ? 'Assessment recorded' : 'Awaiting assessment'}</small></span>
            <span><strong>Grade {row.grade} · {GRADES[row.grade]}</strong><small>{percentage.format(row.severe)} severe DR estimate</small></span>
            <ArrowRight size={18} aria-hidden="true" />
          </button>
        </li>)}</ul>}
    </>}
  </section>
}

const statusLabel: Record<ModelStatus['kind'], string> = {
  checking: 'Checking model', ready: 'Model ready', unavailable: 'Model unavailable', offline: 'API unavailable',
}

function Workspace({ account, onSessionEnd }: { account: Account; onSessionEnd: () => void }) {
  const canCreate = account.role === 'research_admin' || account.role === 'research_operator'
  const canReview = account.role === 'research_admin' || account.role === 'clinician_reviewer'
  const [view, setView] = useState<'queue' | 'new' | 'detail'>('queue')
  const [status, setStatus] = useState<ModelStatus>({ kind: 'checking' })
  const [check, setCheck] = useState(0)
  const [record, setRecord] = useState<CaseRecord | null>(null)
  const [error, setError] = useState('')
  const selection = useRef<AbortController | null>(null)

  useEffect(() => {
    const request = new AbortController()
    void getModelStatus(AbortSignal.any([request.signal, AbortSignal.timeout(6000)])).then((model) => {
      if (!request.signal.aborted) setStatus(model)
    }).catch(() => {
      if (!request.signal.aborted) setStatus({ kind: 'offline' })
    })
    return () => request.abort()
  }, [check])
  useEffect(() => () => selection.current?.abort(), [])

  function show(next: 'queue' | 'new' | 'detail', data: CaseRecord | null = null) {
    selection.current?.abort()
    setError('')
    setRecord(data)
    setView(next)
  }

  async function openCase(id: string) {
    show('detail')
    const request = new AbortController()
    selection.current = request
    try {
      const selected = await readCase(id, AbortSignal.any([request.signal, AbortSignal.timeout(12_000)]))
      if (!request.signal.aborted) setRecord(selected)
    } catch (cause) {
      if (request.signal.aborted) return
      if (sessionEnded(cause)) onSessionEnd()
      else setError(errorText(cause, 'The case could not be loaded.'))
    }
  }

  function recheck() {
    setStatus({ kind: 'checking' })
    setCheck((value) => value + 1)
  }

  return <main className={`shell workspace-main${account.role === 'research_viewer' ? ' viewer-workspace' : ''}`} id="main" tabIndex={-1}>
    <div className="workspace-intro">
      <div><h1>Screening workspace</h1>
        <p className="page-intro">Analyze fundus images and review saved research cases.</p></div>
      <div className="service-status"><span className={`status-pill status-${status.kind}`} role="status">{statusLabel[status.kind]}</span>
        <button className="icon-button" onClick={recheck} type="button" aria-label="Recheck service status" title="Recheck service status"
          disabled={status.kind === 'checking'}><RefreshCw size={16} aria-hidden="true" /></button></div>
    </div>
    <aside className="research-note"><ShieldCheck size={20} aria-hidden="true" />
      <p>Use authorized research images. Predictions and suggested actions are provisional and require clinical judgment.</p></aside>
    <nav className="workspace-nav" aria-label="Workspace views">
      <button className="button button-secondary" type="button" aria-current={view === 'queue' ? 'page' : undefined}
        onClick={() => show('queue')}><ClipboardList size={17} aria-hidden="true" />Case queue</button>
      {canCreate && <button className="button button-primary" type="button" aria-current={view === 'new' ? 'page' : undefined}
        onClick={() => show('new')}><Plus size={17} aria-hidden="true" />New case</button>}
    </nav>
    {view === 'queue' && <CaseQueue onOpen={openCase} onSessionEnd={onSessionEnd} />}
    {view === 'new' && canCreate && <NewCase status={status} onCreate={(created) => show('detail', created)} onSessionEnd={onSessionEnd} />}
    {view === 'detail' && <>
      <button className="button button-text back-to-queue" type="button" onClick={() => show('queue')}>
        <ArrowLeft size={17} aria-hidden="true" />Back to queue</button>
      {error && <p className="analysis-error" role="alert">{error}</p>}
      {!record && !error && <p className="queue-empty" role="status">Loading case…</p>}
      {record && <CaseDetail key={record.id} record={record} canReview={canReview} canExport={account.role !== 'research_viewer'}
        onUpdate={setRecord} onSessionEnd={onSessionEnd} />}
    </>}
  </main>
}

type Session = { status: 'checking' | 'signed_out' | 'signing_out' | 'signout_failed'; notice?: string }
  | { status: 'signed_in'; account: Account }

const sessionChannelName = 'qsart-x-session'

function announceSessionChange() {
  if (typeof BroadcastChannel === 'undefined') return
  const channel = new BroadcastChannel(sessionChannelName)
  channel.postMessage('changed')
  channel.close()
}

function App() {
  const [session, setSession] = useState<Session>({ status: 'checking' })
  const endSession = useCallback(() => setSession({ status: 'signed_out', notice: 'Your session ended. Sign in to continue.' }), [])
  const signedIn = session.status === 'signed_in'

  useEffect(() => {
    if (!signedIn || typeof BroadcastChannel === 'undefined') return
    const channel = new BroadcastChannel(sessionChannelName)
    channel.onmessage = () => setSession({ status: 'checking' })
    return () => channel.close()
  }, [signedIn])

  useEffect(() => {
    if (session.status !== 'checking' && !signedIn) return
    const request = new AbortController()
    async function refresh() {
      try {
        const account = await getSession(AbortSignal.any([request.signal, AbortSignal.timeout(8000)]))
        if (!request.signal.aborted) setSession({ status: 'signed_in', account })
      } catch (cause) {
        if (request.signal.aborted) return
        if (sessionEnded(cause)) {
          if (signedIn) endSession()
          else setSession({ status: 'signed_out' })
        } else if (!signedIn) {
          setSession({ status: 'signed_out', notice: errorText(cause, 'The local portal could not be reached.') })
        }
      }
    }
    function refreshIfVisible() {
      if (document.visibilityState === 'visible') void refresh()
    }
    if (!signedIn) void refresh()
    const timer = signedIn ? window.setInterval(refreshIfVisible, 30_000) : undefined
    if (signedIn) document.addEventListener('visibilitychange', refreshIfVisible)
    return () => {
      request.abort()
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', refreshIfVisible)
    }
  }, [session.status, signedIn, endSession])

  function acceptSignIn(account: Account) {
    setSession({ status: 'signed_in', account })
    announceSessionChange()
  }

  async function logout() {
    setSession({ status: 'signing_out' })
    try {
      await signOut(AbortSignal.timeout(8000))
      setSession({ status: 'signed_out' })
      announceSessionChange()
    } catch (cause) {
      setSession(sessionEnded(cause) ? { status: 'signed_out' } : { status: 'signout_failed' })
    }
  }

  if (session.status === 'signed_in') return <>
    <Header account={session.account} onSignOut={logout} />
    <Workspace key={`${session.account.role}:${session.account.username}`} account={session.account} onSessionEnd={endSession} />
  </>

  return <>
    <Header />
    {session.status === 'signed_out' ? <SignIn notice={session.notice ?? ''} onSignIn={acceptSignIn} /> :
      <main id="main" className="shell workspace-main session-message" tabIndex={-1}>
        {session.status === 'signout_failed' ? <>
          <p className="analysis-error" role="alert">The server could not confirm sign-out.</p>
          <button className="button button-secondary" type="button" onClick={logout}>Retry sign-out</button>
        </> : <p role="status">{session.status === 'checking' ? 'Checking local session…' : 'Signing out…'}</p>}
      </main>}
  </>
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
