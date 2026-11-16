import { StrictMode, useEffect, useState, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from 'react-router'
import {
  ArrowDown, ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight, BookOpen, Camera, CircleHelp,
  ClipboardCheck, Eye, FlaskConical, Focus, GitBranch, Github, ImagePlus, Layers3, Menu,
  ScanEye, Server, ShieldCheck, SlidersHorizontal, X,
} from 'lucide-react'
import { datasets, evidence, pageDetails, roadmap, screeningActions } from './content'
import '@fontsource-variable/inter'
import '@fontsource-variable/manrope'
import './style.css'

const repository = 'https://github.com/Riteshhai/Retina-Opthalmology'
const pages = [
  { path: '/', label: 'Home' },
  { path: '/about', label: 'About' },
  { path: '/research', label: 'Research' },
  { path: '/architecture', label: 'Architecture' },
  { path: '/publications', label: 'Publications' },
  { path: '/contact', label: 'Contact' },
  { path: '/responsible-ai', label: 'Responsible AI' },
  { path: '/demo', label: 'Research demo' },
] as const

function Brand() {
  return (
    <Link className="brand" to="/" aria-label="QSART-X home">
      <img src="/mark.svg" width="38" height="38" alt="" />
      <span>QSART<span className="brand-accent">-X</span></span>
    </Link>
  )
}

function Header() {
  const [open, setOpen] = useState(false)
  const [menuPath, setMenuPath] = useState('')
  const location = useLocation()
  const expanded = open && location.pathname === menuPath
  function toggle() {
    setMenuPath(location.pathname)
    setOpen(!expanded)
  }
  const allLinks = pages.slice(0, -1)
  return (
    <header className="site-header">
      <div className="shell header-inner">
        <Brand />
        <nav id="primary-nav" aria-label="Primary" className={expanded ? 'primary-nav is-open' : 'primary-nav'}>
          {allLinks.map(({ path, label }) => (
            <NavLink key={path} to={path} onClick={() => setOpen(false)} end>{label}</NavLink>
          ))}
          <Link className="button button-primary mobile-demo" to="/demo" onClick={() => setOpen(false)}>
            Research demo <ArrowUpRight size={16} />
          </Link>
        </nav>
        <Link className="button button-primary header-demo" to="/demo">
          Research demo <ArrowUpRight size={15} />
        </Link>
        <button className="menu-toggle" type="button" aria-controls="primary-nav" aria-expanded={expanded}
          aria-label={expanded ? 'Close navigation' : 'Open navigation'} onClick={toggle}>
          {expanded ? <X size={23} /> : <Menu size={23} />}
        </button>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="shell footer-inner">
        <div>
          <Brand />
          <p>Retinal screening research · M.Tech AI &amp; ML, VIT Vellore</p>
        </div>
        <nav aria-label="Footer">
          <Link to="/responsible-ai">Responsible AI &amp; Privacy</Link>
          <Link to="/contact">Contact &amp; Team</Link>
          <a href={repository} aria-label="QSART GitHub repository"><Github size={18} aria-hidden="true" /></a>
        </nav>
      </div>
    </footer>
  )
}

function SectionHeading({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <div className="section-heading">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h2>{title}</h2>
      {children && <p className="heading-detail">{children}</p>}
    </div>
  )
}

const illustrationModes = [
  {
    name: 'Image quality', icon: Focus, marker: 'Visible retinal field',
    description: 'Blur, artifacts and incomplete coverage can obscure features needed for grading.',
  },
  {
    name: 'Disease severity', icon: Eye, marker: 'Retinal structures',
    description: 'A grading model estimates five severity levels from the features visible in the photograph.',
  },
  {
    name: 'Uncertainty', icon: CircleHelp, marker: 'Insufficient evidence',
    description: 'Uncertain estimates can trigger review. A confidence score alone does not qualify a case for automatic clearance.',
  },
] as const

function RetinaIllustration() {
  const [selected, setSelected] = useState(0)
  const mode = illustrationModes[selected]
  return (
    <figure className="retina-card" aria-label="Interactive retinal screening illustration">
      <div className="retina-head flex items-center justify-between gap-3">
        <span className="flex items-center gap-2"><ScanEye size={19} aria-hidden="true" /> Fundus view</span>
        <span className="figure-tag">Simulated illustration</span>
      </div>
      <div className={`retina-stage mode-${selected}`}>
        <div className="scan-corner top-left" /><div className="scan-corner top-right" />
        <div className="scan-corner bottom-left" /><div className="scan-corner bottom-right" />
        <div className="retina-image">
          <img src="/retina-illustration.webp" width="1000" height="1000"
            alt="Simulated color fundus illustration with retinal vessels and optic disc" />
          <span className="retina-region" aria-hidden="true" />
        </div>
        <span className="retina-marker" aria-hidden="true">
          <span />{mode.marker}
        </span>
      </div>
      <div className="retina-controls" role="group" aria-label="Screening illustration topics">
        {illustrationModes.map(({ name, icon: Icon }, index) => (
          <button key={name} type="button" aria-pressed={index === selected} onClick={() => setSelected(index)}>
            <Icon size={15} aria-hidden="true" /> {name}
          </button>
        ))}
      </div>
      <figcaption aria-live="polite">{mode.description}</figcaption>
    </figure>
  )
}

const actionIcons = [ClipboardCheck, Camera, ShieldCheck]

function ScreeningActions() {
  return (
    <div className="action-grid">
      {screeningActions.map(({ code, name, body }, index) => {
        const Icon = actionIcons[index]
        return (
          <article key={code} className={`action-card action-${code.toLowerCase()}`}>
            <header className="flex items-center justify-between gap-4">
              <span className="action-icon"><Icon size={22} aria-hidden="true" /></span>
              <span className="action-code">{code}</span>
            </header>
            <h3>{name}</h3>
            <p>{body}</p>
          </article>
        )
      })}
    </div>
  )
}

function Datasets() {
  return (
    <section className="section band-bordered">
      <div className="shell split-section">
        <SectionHeading eyebrow="Data & methods" title="Know where the evidence comes from">
          Each dataset has a defined role. Training, calibration and evaluation use separate groups where patient identifiers are available.
        </SectionHeading>
        <ul className="dataset-list">
          {datasets.map(({ name, role, status, href }) => (
            <li key={name}>
              <a href={href}>{name}<ArrowUpRight size={16} aria-hidden="true" /></a>
              <p>{role}</p>
              <small>{status}</small>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function Home() {
  return (
    <>
      <section className="home-hero">
        <div className="shell hero-grid">
          <div className="hero-copy">
            <p className="eyebrow hero-eyebrow"><span /> Retinal screening research</p>
            <h1>Smarter retinal screening.<br /><span>Safer decisions.</span></h1>
            <p className="hero-intro">
              QSART-X studies how image quality, disease severity and uncertainty can guide retinal screening
              when repeat imaging and specialist review have separate capacity limits.
            </p>
            <div className="flex flex-wrap items-center gap-3 hero-actions">
              <Link className="button button-primary" to="/demo">Explore the demo <ArrowRight size={17} /></Link>
              <Link className="button button-secondary" to="/research">Read the research <ArrowUpRight size={16} /></Link>
            </div>
            <p className="hero-note"><ShieldCheck size={15} aria-hidden="true" /> Research prototype for retrospective evaluation</p>
          </div>
          <RetinaIllustration />
        </div>
      </section>
      <section className="evidence-band" aria-labelledby="recorded-results">
        <div className="shell">
          <div className="evidence-heading">
            <h2 id="recorded-results" className="eyebrow">Recorded study results</h2>
            <Link className="text-link" to="/publications">Study reports <ArrowUpRight size={15} /></Link>
          </div>
          <dl className="evidence-grid">
            {evidence.map(({ value, label, context }) => (
              <div className="evidence-item" key={label}>
                <dt>{label}</dt><dd className="metric-value">{value}</dd><dd className="metric-context">{context}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>
      <section className="section">
        <div className="shell">
          <div className="section-top">
            <SectionHeading eyebrow="Screening workflow" title="Three actions, two resource limits">
              The decision engine accounts for repeat imaging and specialist review capacity separately.
            </SectionHeading>
            <Link className="text-link" to="/architecture">See the architecture <ArrowUpRight size={16} /></Link>
          </div>
          <ScreeningActions />
        </div>
      </section>
      <section className="section dark-band">
        <div className="shell split-section">
          <SectionHeading eyebrow="Research question" title="What would another view add?">
            QSART-X tests whether the expected information from an additional retinal view can justify acquisition at a fixed workload.
          </SectionHeading>
          <div className="research-aside">
            <ImagePlus size={28} aria-hidden="true" />
            <p>Paired photographs support retrospective comparison of acquisition policies. The acquisition rule uses only the first view to decide whether to request the second.</p>
            <Link className="text-link" to="/research">Research methods <ArrowRight size={16} /></Link>
          </div>
        </div>
      </section>
      <Datasets />
      <section className="section">
        <div className="shell split-section">
          <SectionHeading eyebrow="Project roadmap" title="From retrospective studies to a research workspace">
            M.Tech in Artificial Intelligence and Machine Learning, VIT Vellore.
          </SectionHeading>
          <ol className="roadmap">
            {roadmap.map(({ title, body, status }) => (
              <li key={title}><div><h3>{title}</h3><span>{status}</span></div><p>{body}</p></li>
            ))}
          </ol>
        </div>
      </section>
    </>
  )
}

function PageHero({ eyebrow, title, intro }: { eyebrow: string; title: string; intro: string }) {
  return (
    <div className="page-hero">
      <div className="shell">
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p>{intro}</p>
      </div>
    </div>
  )
}

const detailIcons = {
  about: Eye,
  research: FlaskConical,
  publications: BookOpen,
  contact: Github,
  'responsible-ai': ShieldCheck,
}

function EditorialPage({ page }: { page: keyof typeof pageDetails }) {
  const { sections, ...intro } = pageDetails[page]
  const Icon = detailIcons[page]
  return (
    <>
      <PageHero {...intro} />
      <section className="section">
        <div className="shell editorial-layout">
          <nav className="editorial-index" aria-label="On this page">
            <Icon size={24} aria-hidden="true" />
            <ul>{sections.map(({ title }, index) => <li key={title}><a href={`#section-${index + 1}`}>{title}</a></li>)}</ul>
          </nav>
          <div className="editorial-sections">
            {sections.map(({ title, body, links }, index) => (
              <article id={`section-${index + 1}`} key={title}>
                <h2>{title}</h2><p>{body}</p>
                {links && <ul className="source-links">{links.map(({ label, href }) => (
                  <li key={href}><a href={href}>{label}<ArrowUpRight size={16} aria-hidden="true" /></a></li>
                ))}</ul>}
              </article>
            ))}
          </div>
        </div>
      </section>
      {page === 'research' && <Datasets />}
    </>
  )
}

const stages = {
  input: { name: 'Fundus input', icon: Camera, body: 'A color fundus photograph provides the image for analysis.' },
  processing: { name: 'Image processing', icon: Focus, body: 'The pipeline checks the image format and prepares the orientation, crop, size and RGB pixel values required by the model.' },
  quality: { name: 'Quality estimation', icon: ScanEye, body: 'The quality model estimates overall gradability. Additional heads assess visible defects using source-specific labels.' },
  severity: { name: 'Severity estimation', icon: Layers3, body: 'The grading model estimates probabilities for five diabetic retinopathy grades. Grades 2 and above define the project’s referability endpoint.' },
  uncertainty: { name: 'Uncertainty estimation', icon: CircleHelp, body: 'The pipeline summarizes predictive uncertainty and can generate a set of plausible grades using source-calibrated conformal methods.' },
  decision: { name: 'Decision engine', icon: SlidersHorizontal, body: 'The workflow combines severity, quality and uncertainty criteria under independent budgets for repeat imaging and specialist review.' },
  action: { name: 'Screening action', icon: ClipboardCheck, body: 'The policy can recommend automated non-referable disposition, another image or specialist review. It reports overload when required deferrals exceed capacity.' },
  second: { name: 'Second-view acquisition', icon: ImagePlus, body: 'This proposed extension estimates the benefit of acquiring another view, based on the first view and the cost of acquisition. Retrospective paired photographs provide the evaluation input.' },
} satisfies Record<string, { name: string; icon: typeof Camera; body: string }>

type Stage = keyof typeof stages

function Architecture() {
  const [active, setActive] = useState<Stage>('input')
  function node(stage: Stage) {
    const { name, icon: Icon } = stages[stage]
    return (
      <button key={stage} className={`stage-node ${stage === 'second' ? 'stage-proposed' : ''}`}
        type="button" aria-pressed={stage === active} onClick={() => setActive(stage)}>
        <Icon size={21} aria-hidden="true" />
        <span>{name}{stage === 'second' && <small>Proposed extension</small>}</span>
      </button>
    )
  }
  const Icon = stages[active].icon
  return (
    <>
      <PageHero eyebrow="Architecture" title="From retinal image to screening action"
        intro="Select a stage to see how the pipeline turns a fundus photograph into an experimental workflow recommendation." />
      <section className="section">
        <div className="shell architecture-layout">
          <div className="architecture-panel" role="group" aria-label="Screening architecture stages">
            <div className="stage-row stage-linear">{node('input')}<ArrowRight size={20} aria-hidden="true" />{node('processing')}</div>
            <ArrowDown className="stage-arrow" size={20} aria-hidden="true" />
            <div className="stage-row stage-parallel">{node('quality')}{node('severity')}{node('uncertainty')}</div>
            <ArrowDown className="stage-arrow" size={20} aria-hidden="true" />
            <div className="stage-row stage-linear">{node('decision')}<ArrowRight size={20} aria-hidden="true" />{node('action')}</div>
            <div className="stage-extension"><ArrowDownRight size={20} aria-hidden="true" />{node('second')}</div>
          </div>
          <aside className="architecture-detail" aria-live="polite">
            <span className="action-icon"><Icon size={24} aria-hidden="true" /></span>
            <h2>{stages[active].name}</h2><p>{stages[active].body}</p>
          </aside>
        </div>
      </section>
    </>
  )
}

type ServiceState = { health: 'checking' | 'online' | 'offline'; model: 'checking' | 'ready' | 'unavailable' }

async function readStatus(endpoint: 'health' | 'ready', signal: AbortSignal) {
  const response = await fetch(`/api/${endpoint}`, { signal, cache: 'no-store' })
  if (!response.headers.get('content-type')?.includes('application/json')) return false
  const data: unknown = await response.json()
  if (!data || typeof data !== 'object') return false
  return response.ok && (endpoint === 'health'
    ? 'status' in data && data.status === 'ok'
    : 'ready' in data && data.ready === true)
}

function ServiceStatus() {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<ServiceState>({ health: 'checking', model: 'checking' })
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const timer = window.setTimeout(() => controller.abort(), 4500)
    void Promise.allSettled([readStatus('health', controller.signal), readStatus('ready', controller.signal)])
      .then(([healthResult, readyResult]) => {
        if (!active) return
        const online = healthResult.status === 'fulfilled' && healthResult.value
        const ready = readyResult.status === 'fulfilled' && readyResult.value
        setState({ health: online ? 'online' : 'offline', model: ready ? 'ready' : 'unavailable' })
      }).finally(() => window.clearTimeout(timer))
    return () => {
      active = false
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [attempt])
  function retry() {
    setState({ health: 'checking', model: 'checking' })
    setAttempt((previous) => previous + 1)
  }
  const healthLabel = { checking: 'Checking…', online: 'Online', offline: 'Unavailable' }[state.health]
  const modelLabel = { checking: 'Checking…', ready: 'Ready', unavailable: 'Unavailable' }[state.model]
  return (
    <aside className="status-card">
      <span className="action-icon"><Server size={23} aria-hidden="true" /></span>
      <h2>Local research service</h2>
      <dl className="status-list" aria-live="polite">
        <div><dt>API connection</dt><dd className={state.health}>{healthLabel}</dd></div>
        <div><dt>Model artifacts</dt><dd className={state.model}>{modelLabel}</dd></div>
      </dl>
      <p>The public demo explains the workflow. Authorized inference runs in the local research workspace with trained artifacts.</p>
      <button className="button button-secondary" type="button" onClick={retry} disabled={state.health === 'checking'}>
        Recheck status <ArrowRight size={16} aria-hidden="true" />
      </button>
    </aside>
  )
}

function Demo() {
  return (
    <>
      <PageHero eyebrow="Research demo" title="Explore the screening workflow"
        intro="Use the simulated fundus illustration to explore the three measurements that inform a triage recommendation." />
      <section className="section">
        <div className="shell demo-layout">
          <RetinaIllustration />
          <div className="demo-side">
            <ServiceStatus />
            <Link to="/architecture" className="text-link">Follow the architecture <GitBranch size={17} aria-hidden="true" /></Link>
          </div>
        </div>
      </section>
    </>
  )
}

function NotFound() {
  return (
    <div className="shell section not-found">
      <p className="eyebrow">Page unavailable</p>
      <h1>We couldn’t find that page.</h1>
      <Link className="text-link" to="/"><ArrowLeft size={16} /> Back to home</Link>
    </div>
  )
}

function App() {
  const { pathname } = useLocation()
  useEffect(() => {
    const label = pages.find((page) => page.path === pathname)?.label
    document.title = pathname === '/' ? 'QSART-X · Retinal screening research' : `QSART-X · ${label ?? 'Page unavailable'}`
    window.scrollTo(0, 0)
  }, [pathname])
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <main id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/demo" element={<Demo />} />
          {Object.keys(pageDetails).map((page) => (
            <Route key={page} path={`/${page}`} element={<EditorialPage page={page as keyof typeof pageDetails} />} />
          ))}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
    </>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode><BrowserRouter><App /></BrowserRouter></StrictMode>,
)
