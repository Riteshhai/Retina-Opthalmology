import {
  isAction, isGrade, isProbability, isRecord, isTimestamp, parsePrediction,
  type AnalysisResult, type Upload,
} from './inference.ts'

export const ROLES = {
  research_admin: 'Research administrator',
  research_operator: 'Research operator',
  clinician_reviewer: 'Clinician / reviewer',
  research_viewer: 'Read-only research viewer',
} as const

type Action = AnalysisResult['action']
export type Account = { username: string; role: keyof typeof ROLES }
export type CaseSummary = {
  id: string
  createdAt: string
  grade: number
  severe: number
  modelAction: Action
  effectiveAction: Action
  reviewed: boolean
}
export type ReviewInput = { grade: number; action: Action | null; notes: string }
type Review = ReviewInput & {
  id: string; reviewer: string; createdAt: string; gradeDisagreement: boolean; isOverride: boolean
}
type CaseEvent = { event: string; actor: string; createdAt: string }
export type CaseRecord = {
  id: string; createdAt: string; prediction: AnalysisResult; reviews: Review[];
  effectiveAction: Action; audit: CaseEvent[]
}

export class PortalError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.status = status
  }
}

const commonErrors: Record<number, string> = {
  401: 'Your session has ended. Sign in again to continue.',
  403: 'Your account does not have permission for this action.',
  404: 'This case is unavailable.',
  429: 'Too many attempts. Wait a short time before trying again.',
  503: 'The local portal is unavailable. Check the service and try again.',
}

async function request(path: string, signal: AbortSignal, body?: BodyInit, errors?: Record<number, string>) {
  const headers = body === undefined ? undefined : new Headers({ 'X-QSART-Workspace': '1' })
  if (typeof body === 'string' && body.length) headers?.set('Content-Type', 'application/json')
  const response = await fetch(`/api/portal${path}`, {
    signal, cache: 'no-store', credentials: 'same-origin', method: body === undefined ? 'GET' : 'POST', body, headers,
  })
  if (!response.ok) throw new PortalError(errors?.[response.status] ?? commonErrors[response.status]
    ?? 'The local portal could not complete this request.', response.status)
  return response
}

function parseAccount(value: unknown): Account {
  if (!isRecord(value) || typeof value.username !== 'string'
      || typeof value.role !== 'string' || !Object.hasOwn(ROLES, value.role)) {
    throw new PortalError('The portal returned an invalid account response.')
  }
  return { username: value.username, role: value.role as Account['role'] }
}

export async function getSession(signal: AbortSignal) {
  return parseAccount(await (await request('/session', signal)).json())
}

export async function signIn(username: string, password: string, signal: AbortSignal) {
  return parseAccount(await (await request('/login', signal, JSON.stringify({ username, password }), {
    401: 'The username or password was not accepted.',
    422: 'Enter a username and password.',
  })).json())
}

export async function signOut(signal: AbortSignal) {
  await request('/logout', signal, '')
}

function parseSummary(value: unknown): CaseSummary {
  if (!isRecord(value) || typeof value.id !== 'string' || !isTimestamp(value.created_at)
      || !isGrade(value.predicted_grade) || !isProbability(value.severe_probability)
      || !isAction(value.model_action) || !isAction(value.effective_action) || typeof value.reviewed !== 'boolean') {
    throw new PortalError('The portal returned an invalid case list.')
  }
  return { id: value.id, createdAt: value.created_at, grade: value.predicted_grade, severe: value.severe_probability,
    modelAction: value.model_action, effectiveAction: value.effective_action, reviewed: value.reviewed }
}

function parseReview(value: unknown): Review {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.reviewer !== 'string'
      || !isGrade(value.grade) || (value.action !== null && !isAction(value.action))
      || typeof value.notes !== 'string' || !isTimestamp(value.created_at)
      || typeof value.grade_disagreement !== 'boolean' || typeof value.is_override !== 'boolean') {
    throw new PortalError('The portal returned an invalid review record.')
  }
  return { id: value.id, reviewer: value.reviewer, grade: value.grade, notes: value.notes, action: value.action,
    createdAt: value.created_at, gradeDisagreement: value.grade_disagreement, isOverride: value.is_override }
}

function parseEvent(value: unknown): CaseEvent {
  if (!isRecord(value) || typeof value.event !== 'string' || typeof value.actor !== 'string'
      || !isTimestamp(value.created_at)) throw new PortalError('The portal returned an invalid action history.')
  return { event: value.event, actor: value.actor, createdAt: value.created_at }
}

function parseCase(value: unknown): CaseRecord {
  if (!isRecord(value) || typeof value.id !== 'string' || !isTimestamp(value.created_at)
      || !isAction(value.effective_action) || !Array.isArray(value.reviews) || !Array.isArray(value.audit)) {
    throw new PortalError('The portal returned an invalid case record.')
  }
  return { id: value.id, createdAt: value.created_at, prediction: parsePrediction(value.prediction),
    reviews: value.reviews.map(parseReview), effectiveAction: value.effective_action, audit: value.audit.map(parseEvent) }
}

export async function listCases(signal: AbortSignal) {
  const value: unknown = await (await request('/cases', signal)).json()
  if (!Array.isArray(value)) throw new PortalError('The portal returned an invalid case list.')
  return value.map(parseSummary)
}

const casePath = (id: string) => `/cases/${encodeURIComponent(id)}`

export async function readCase(id: string, signal: AbortSignal) {
  return parseCase(await (await request(casePath(id), signal)).json())
}

export async function readImage(id: string, signal: AbortSignal) {
  const response = await request(`${casePath(id)}/image`, signal)
  if (response.headers.get('Content-Type') !== 'image/png') throw new PortalError('The saved photograph is unavailable.')
  return response.blob()
}

export async function createCase(upload: Upload, signal: AbortSignal): Promise<CaseRecord> {
  const form = new FormData()
  const filename = upload.type === 'image/png' ? 'fundus.png' : 'fundus.jpg'
  form.append('file', upload.file.slice(0, upload.file.size, upload.type), filename)
  const response = await request('/cases', signal, form, {
    413: 'The file exceeds the upload limit.',
    415: 'The image content must match its JPG or PNG format.',
    422: 'The image could not be read or its dimensions are unsupported.',
    503: 'The model or case storage is unavailable. Recheck the service before trying again.',
  })
  return parseCase(await response.json())
}

export async function saveReview(id: string, review: ReviewInput, signal: AbortSignal) {
  return parseCase(await (await request(`${casePath(id)}/reviews`, signal, JSON.stringify(review), {
    422: 'Select a grade and provide notes for any action override.',
  })).json())
}

export async function downloadReport(id: string, signal: AbortSignal) {
  const response = await request(`${casePath(id)}/report`, signal)
  return response.blob()
}

export function sessionEnded(cause: unknown) {
  return cause instanceof PortalError && cause.status === 401
}

export function errorText(cause: unknown, fallback: string) {
  return cause instanceof PortalError ? cause.message : fallback
}
