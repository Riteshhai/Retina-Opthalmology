export const MIME_TYPES = ['image/jpeg', 'image/png'] as const
type ImageType = typeof MIME_TYPES[number]
export const MAX_UPLOAD_BYTES = 12 * 1024 * 1024

export const ACTIONS = {
  AUTO_NONREFERABLE: 'Provisional non-referable disposition',
  RECAPTURE: 'Request another image',
  SPECIALIST_REVIEW: 'Specialist review',
} as const

export const GRADES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'] as const

export type ModelStatus = { kind: 'checking' | 'offline' | 'unavailable' } | {
  kind: 'ready'
  version: string
  qualityVersion: string
  device: string
  maxBytes: number
  types: ImageType[]
}

export type Upload = { file: File; type: ImageType }

export type AnalysisResult = {
  action: keyof typeof ACTIONS
  predictedGrade: number
  gradeProbabilities: number[]
  referable: number
  severe: number
  poorQuality: number
  entropy: number
  confidence: number
  modelVersion: string
  policyVersion: string
  timestamp: string
  processingTime: number
  reasons: string[]
  conformalGrades: number[] | null
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isProbability(value: unknown): value is number {
  return typeof value === 'number' && value >= 0 && value <= 1
}

export function isGrade(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < GRADES.length
}

export function isAction(value: unknown): value is keyof typeof ACTIONS {
  return typeof value === 'string' && Object.hasOwn(ACTIONS, value)
}

export function isTimestamp(value: unknown): value is string {
  return typeof value === 'string' && Number.isFinite(Date.parse(value))
}

export function parseModelStatus(value: unknown): ModelStatus {
  if (!isRecord(value)) return { kind: 'offline' }
  if (value.ready === false) return { kind: 'unavailable' }
  if (value.ready !== true || value.data_purpose !== 'real_research' || typeof value.model_version !== 'string'
      || typeof value.quality_model_version !== 'string' || typeof value.device !== 'string'
      || typeof value.max_upload_bytes !== 'number' || !Number.isSafeInteger(value.max_upload_bytes)
      || value.max_upload_bytes < 1 || !Array.isArray(value.supported_types)) return { kind: 'offline' }
  const reportedTypes: unknown[] = value.supported_types
  const types = MIME_TYPES.filter((type) => reportedTypes.includes(type))
  if (!types.length) return { kind: 'offline' }
  return { kind: 'ready', version: value.model_version, qualityVersion: value.quality_model_version,
    device: value.device, maxBytes: Math.min(value.max_upload_bytes, MAX_UPLOAD_BYTES), types }
}

export function validateUpload(file: File, types: readonly ImageType[], limit: number): Upload {
  const type = file.type || (/\.png$/i.test(file.name) ? 'image/png' : /\.jpe?g$/i.test(file.name) ? 'image/jpeg' : '')
  const accepted = types.find((value) => value === type)
  if (!accepted) throw new Error('Choose a JPG or PNG file.')
  if (file.size === 0) throw new Error('The selected file is empty.')
  if (file.size > limit) throw new Error(`The file exceeds the ${+(limit / (1024 * 1024)).toFixed(2)} MiB limit.`)
  return { file, type: accepted }
}

export function parsePrediction(value: unknown): AnalysisResult {
  if (!isRecord(value) || value.policy_scope !== 'single_image' || !isRecord(value.quality)
      || !isRecord(value.uncertainty) || !isGrade(value.predicted_grade)
      || !Array.isArray(value.grade_probabilities) || value.grade_probabilities.length !== GRADES.length
      || !value.grade_probabilities.every(isProbability)
      || !isProbability(value.referable_probability) || !isProbability(value.severe_probability)
      || !isProbability(value.quality.poor_probability) || !isProbability(value.uncertainty.entropy)
      || !isProbability(value.uncertainty.confidence)
      || typeof value.model_version !== 'string' || typeof value.policy_version !== 'string'
      || !isAction(value.action) || !isTimestamp(value.inference_timestamp)
      || typeof value.processing_time_ms !== 'number' || !Number.isFinite(value.processing_time_ms)
      || value.processing_time_ms < 0 || !Array.isArray(value.reason_codes)
      || !value.reason_codes.every((reason) => typeof reason === 'string')) {
    throw new PredictionError('The API returned an incomplete analysis. Try again after checking service status.')
  }
  let conformalGrades: number[] | null = null
  if (isRecord(value.conformal) && value.conformal.status === 'available') {
    if (!Array.isArray(value.conformal.prediction_set) || !value.conformal.prediction_set.every(isGrade)) {
      throw new PredictionError('The API returned an invalid conformal grade set.')
    }
    conformalGrades = value.conformal.prediction_set
  }
  return { action: value.action, predictedGrade: value.predicted_grade,
    gradeProbabilities: value.grade_probabilities, referable: value.referable_probability,
    severe: value.severe_probability, poorQuality: value.quality.poor_probability,
    entropy: value.uncertainty.entropy, confidence: value.uncertainty.confidence,
    modelVersion: value.model_version, policyVersion: value.policy_version, timestamp: value.inference_timestamp,
    processingTime: value.processing_time_ms, reasons: value.reason_codes, conformalGrades }
}

export async function getModelStatus(signal: AbortSignal): Promise<ModelStatus> {
  const response = await fetch('/api/model/info', { signal, cache: 'no-store' })
  return response.ok ? parseModelStatus(await response.json()) : { kind: 'offline' }
}

export class PredictionError extends Error {
}
