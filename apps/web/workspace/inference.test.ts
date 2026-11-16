import assert from 'node:assert/strict'
import test from 'node:test'
import {
  MAX_UPLOAD_BYTES, MIME_TYPES, analyzeImage, parseModelStatus, parsePrediction, validateUpload,
} from './inference.ts'

const softwareFixture = {
  action: 'SPECIALIST_REVIEW', policy_scope: 'single_image', model_version: 'software-test-fixture',
  policy_version: '2.0', predicted_grade: 2, grade_probabilities: [.1, .05, .55, .2, .1],
  referable_probability: .85, severe_probability: .3, quality: { poor_probability: .12 },
  uncertainty: { entropy: .7, confidence: .55 },
  inference_timestamp: '2026-01-01T00:00:00Z', processing_time_ms: 5,
  reason_codes: ['PREDICTED_REFERABLE_GRADE'], conformal: { status: 'available', prediction_set: [1, 2, 3] },
}

test('distinguishes model artifacts, software fixtures and invalid status bodies', () => {
  const info = { ready: true, data_purpose: 'real_research', model_version: 'contract-fixture-dr',
    quality_model_version: 'contract-fixture-quality', device: 'cpu', max_upload_bytes: MAX_UPLOAD_BYTES,
    supported_types: MIME_TYPES }
  assert.equal(parseModelStatus(info).kind, 'ready')
  assert.deepEqual(parseModelStatus({ ready: false }), { kind: 'unavailable' })
  assert.deepEqual(parseModelStatus({ ...info, data_purpose: 'software_test' }), { kind: 'offline' })
  assert.deepEqual(parseModelStatus({ ...info, supported_types: ['text/plain'] }), { kind: 'offline' })
  assert.deepEqual(parseModelStatus(null), { kind: 'offline' })
})

test('validates the size and media type of a selected upload', () => {
  assert.equal(validateUpload(new File(['fixture'], 'image.jpg'), MIME_TYPES, MAX_UPLOAD_BYTES).type, 'image/jpeg')
  assert.throws(() => validateUpload(new File(['fixture'], 'image.txt'), MIME_TYPES, MAX_UPLOAD_BYTES), /JPG or PNG/)
  assert.throws(() => validateUpload(new File([], 'image.jpg'), MIME_TYPES, MAX_UPLOAD_BYTES), /empty/)
  assert.throws(() => validateUpload(new File(['fixture'], 'image.jpg'), MIME_TYPES, 1), /limit/)
})

test('uses the API values and rejects an invalid prediction contract', () => {
  const modelResult = parsePrediction(softwareFixture)
  assert.equal(modelResult.action, 'SPECIALIST_REVIEW')
  assert.equal(modelResult.severe, softwareFixture.severe_probability)
  assert.deepEqual(modelResult.gradeProbabilities, softwareFixture.grade_probabilities)
  assert.deepEqual(modelResult.conformalGrades, [1, 2, 3])
  assert.equal(parsePrediction({ ...softwareFixture, conformal: null }).conformalGrades, null)
  assert.throws(() => parsePrediction({ ...softwareFixture, action: 'UNKNOWN' }), /incomplete/)
  assert.throws(() => parsePrediction({ ...softwareFixture, grade_probabilities: [1] }), /incomplete/)
  assert.throws(() => parsePrediction({ ...softwareFixture, policy_scope: 'cohort' }), /incomplete/)
  assert.throws(() => parsePrediction({ ...softwareFixture, inference_timestamp: 'invalid' }), /incomplete/)
})

test('uploads multipart bytes with a generic filename and surface-level errors', async (context) => {
  let status = 200
  const upload = validateUpload(new File(['fixture'], 'local-research-record.png'), MIME_TYPES, MAX_UPLOAD_BYTES)
  context.mock.method(globalThis, 'fetch', async (url: string, request: RequestInit) => {
    assert.equal(url, '/api/predict')
    assert.equal(request.method, 'POST')
    assert.equal(new Headers(request.headers).get('X-QSART-Workspace'), '1')
    assert.ok(request.body instanceof FormData)
    const file = request.body.get('file')
    assert.ok(file instanceof File)
    assert.equal(file.name, 'fundus.png')
    assert.equal(file.type, 'image/png')
    assert.equal(await file.text(), 'fixture')
    return Response.json(status === 200 ? softwareFixture : { detail: 'Discard this response detail.' }, { status })
  })
  const signal = new AbortController().signal
  assert.equal((await analyzeImage(upload, signal)).predictedGrade, 2)
  status = 503
  await assert.rejects(analyzeImage(upload, signal), /could not complete this analysis/)
})
