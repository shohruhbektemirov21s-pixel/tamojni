// Mock backend — §7 API kontraktini (SSE, multipart, xato envelope, §7.1 result)
// AYNAN taqlid qiladi. Real backend tayyor bo'lguncha parallel ishlash uchun (ADR-005).
// Real backend bilan bir xil HTTP yo'l — app kodi o'zgarmaydi, faqat endpoint config.
//
// Ishga tushirish:  node mock-server/server.mjs   (default port 8787)
// App Sozlamalar'da endpoint = http://127.0.0.1:8787 qilib qo'ying.
import express from 'express'
import multer from 'multer'
import cors from 'cors'

const PORT = process.env.MOCK_PORT ? Number(process.env.MOCK_PORT) : 8787
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } })

const app = express()
app.use(cors())
app.use(express.json())

// ---------------- xotira ombori ----------------
/** @type {Map<string, any>} */
const cases = new Map()
let seq = 1000
let auditSeq = 5000

const now = () => new Date().toISOString()
const err = (res, status, code, message, detail) =>
  res.status(status).json({ error: { code, message, detail: detail ?? null } })

// ---------------- rasm o'lchamini header'dan o'qish (dep yo'q) ----------------
function imageSize(buf) {
  // PNG: 89 50 4E 47 ... IHDR width@16 height@20 (big-endian)
  if (buf.length > 24 && buf[0] === 0x89 && buf[1] === 0x50) {
    return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }
  }
  // JPEG: SOFx markerlarini qidiramiz
  if (buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let o = 2
    while (o < buf.length) {
      if (buf[o] !== 0xff) {
        o++
        continue
      }
      const marker = buf[o + 1]
      // SOF0..SOF15 (C4/C8/CC bundan mustasno)
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { h: buf.readUInt16BE(o + 5), w: buf.readUInt16BE(o + 7) }
      }
      const len = buf.readUInt16BE(o + 2)
      o += 2 + len
    }
  }
  return { w: 640, h: 480 } // fallback
}

// ---------------- noisy-OR risk (backend mocks.py reference bilan bir xil) ----------------
function computeRisk(detections) {
  const weights = { pichoq: 0.8, 'qurol-qism': 0.85, suyuqlik: 0.5, 'organik-massa': 0.45 }
  const factors = detections.map((d) => {
    const weight = weights[d.class] ?? 0.2
    const contribution = Math.round(weight * d.confidence * 1000) / 1000
    return { rule: 'prohibited_class', class: d.class, confidence: d.confidence, weight, contribution }
  })
  let prod = 1
  for (const f of factors) prod *= 1 - f.contribution
  const score = Math.round((1 - prod) * 1000) / 1000
  const level = score >= 0.7 ? 'HIGH' : score >= 0.4 ? 'MEDIUM' : 'LOW'
  return { level, score, computed_by: 'rule_engine_ref_v1', factors }
}

// ---------------- demo ssenariylari (barcha UI holatlarini ko'rsatish) ----------------
// Ketma-ket aylanadi: HIGH -> MEDIUM -> LOW(bo'sh) -> DEGRADED
function buildScenario(idx, size) {
  const { w, h } = size
  const box = (fx1, fy1, fx2, fy2) => [
    Math.round(fx1 * w),
    Math.round(fy1 * h),
    Math.round(fx2 * w),
    Math.round(fy2 * h)
  ]
  const s = idx % 4
  if (s === 0) {
    const detections = [
      { class: 'pichoq', confidence: 0.91, bbox: box(0.18, 0.22, 0.42, 0.55) },
      { class: 'qurol-qism', confidence: 0.74, bbox: box(0.55, 0.4, 0.78, 0.66) }
    ]
    return { detections, degraded: false, sttOk: true, llmOk: true }
  }
  if (s === 1) {
    const detections = [{ class: 'suyuqlik', confidence: 0.81, bbox: box(0.35, 0.3, 0.6, 0.7) }]
    return { detections, degraded: false, sttOk: true, llmOk: true }
  }
  if (s === 2) {
    return { detections: [], degraded: false, sttOk: true, llmOk: true } // "topilmadi != xavfsiz"
  }
  // degraded: STT va LLM ishlamadi, lekin detektsiya+risk bor
  const detections = [{ class: 'organik-massa', confidence: 0.63, bbox: box(0.4, 0.35, 0.62, 0.62) }]
  return { detections, degraded: true, sttOk: false, llmOk: false }
}

function finalizeResult(c) {
  const sc = c._scenario
  const risk = computeRisk(sc.detections)
  c.risk = risk
  c.detections = sc.detections
  c.degraded = sc.degraded
  c.transcript = sc.sttOk
    ? { text: 'Yuk hujjatlari to\'liq, deklaratsiya raqami 4521.', language: 'uz', confidence: 0.88, available: true }
    : { text: null, language: null, confidence: null, available: false }
  c.explanation = sc.llmOk
    ? {
        text: `Aniqlangan ${sc.detections.length} ta obyekt asosida risk darajasi ${risk.level} (${risk.score}). Eng katta hissa "${risk.factors[0]?.class ?? '—'}" sinfidan. Operator ko'rib chiqishi tavsiya etiladi.`,
        generated_by: 'mock-qwen3-4b',
        available: true
      }
    : { text: null, generated_by: null, available: false }
  c.timings_ms = { detection: 320, stt: sc.sttOk ? 540 : 0, risk: 4, llm: sc.llmOk ? 1180 : 0 }
  audit(c, 'pipeline', 'DETECTION_DONE', { count: sc.detections.length })
  if (sc.sttOk) audit(c, 'pipeline', 'STT_DONE', {})
  audit(c, 'pipeline', 'RISK_COMPUTED', { level: risk.level, score: risk.score })
  if (sc.llmOk) audit(c, 'pipeline', 'EXPLANATION_DONE', {})
  if (sc.degraded) audit(c, 'pipeline', 'MODEL_FAILED', { models: ['stt', 'llm'] })
}

function audit(c, actor, action, payload) {
  c._audit.push({ id: `aud-${auditSeq++}`, actor, action, payload: payload ?? null, ts: now() })
}

function serialize(c) {
  return {
    case_id: c.case_id,
    status: c.status,
    risk: c.risk,
    detections: c.detections,
    transcript: c.transcript,
    explanation: c.explanation,
    operator_notes: c.operator_notes,
    timings_ms: c.timings_ms,
    degraded: c.degraded
  }
}

// ---------------- pipeline simulyatsiyasi ----------------
function runPipeline(c) {
  setTimeout(() => {
    c.status = 'PROCESSING'
  }, 800)
  setTimeout(() => {
    finalizeResult(c)
    c.status = 'DONE'
  }, 2600)
}

// ==================== ENDPOINTLAR (§7) ====================

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    models: { yolo: 'mock-ready', stt: 'mock-ready', llm: 'mock-ready' },
    gpu: { available: false, name: 'mock-cpu' }
  })
})

app.post('/cases', upload.fields([{ name: 'image', maxCount: 1 }, { name: 'audio', maxCount: 1 }]), (req, res) => {
  const img = req.files?.image?.[0]
  if (!img || !img.buffer?.length) {
    return err(res, 400, 'validation_failed', "Rasm bo'sh yoki yuborilmadi", { field: 'image' })
  }
  const id = `case-${seq++}`
  const idx = seq // scenario rotation
  const c = {
    case_id: id,
    status: 'PENDING',
    risk: null,
    detections: [],
    transcript: null,
    explanation: null,
    operator_notes: req.body?.notes ?? null,
    timings_ms: null,
    degraded: false,
    created_at: now(),
    _audit: [],
    _scenario: buildScenario(idx, imageSize(img.buffer)),
    _decided: false
  }
  audit(c, req.body?.operator_id ?? 'operator', 'CASE_CREATED', {
    has_audio: !!req.files?.audio?.[0]
  })
  cases.set(id, c)
  runPipeline(c)
  res.status(201).json({ case_id: id, status: 'PENDING' })
})

app.get('/cases', (req, res) => {
  const { status, risk_level } = req.query
  const limit = Math.min(500, Math.max(1, Number(req.query.limit) || 50))
  const offset = Math.max(0, Number(req.query.offset) || 0)
  let all = [...cases.values()].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
  if (status) all = all.filter((c) => c.status === status)
  if (risk_level) all = all.filter((c) => c.risk?.level === risk_level)
  const total = all.length
  const items = all.slice(offset, offset + limit).map((c) => ({
    case_id: c.case_id,
    status: c.status,
    risk_level: c.risk?.level ?? null,
    degraded: c.degraded,
    created_at: c.created_at
  }))
  res.json({ items, total })
})

app.get('/cases/:id', (req, res) => {
  const c = cases.get(req.params.id)
  if (!c) return err(res, 404, 'not_found', 'Case topilmadi', { case_id: req.params.id })
  res.json(serialize(c))
})

app.get('/cases/:id/audit', (req, res) => {
  const c = cases.get(req.params.id)
  if (!c) return err(res, 404, 'not_found', 'Case topilmadi', { case_id: req.params.id })
  res.json({ entries: c._audit })
})

app.post('/cases/:id/decision', (req, res) => {
  const c = cases.get(req.params.id)
  if (!c) return err(res, 404, 'not_found', 'Case topilmadi', { case_id: req.params.id })
  const decision = req.body?.decision
  if (!['CONFIRM', 'REJECT', 'OVERRIDE'].includes(decision)) {
    return err(res, 400, 'validation_failed', 'Noto\'g\'ri qaror turi', { field: 'decision' })
  }
  const actionMap = {
    CONFIRM: 'OPERATOR_CONFIRMED',
    REJECT: 'OPERATOR_REJECTED',
    OVERRIDE: 'OPERATOR_OVERRIDE'
  }
  c.operator_notes = req.body?.notes ?? c.operator_notes
  const id = `aud-${auditSeq++}`
  c._audit.push({
    id,
    actor: req.body?.operator_id ?? 'operator',
    action: actionMap[decision],
    payload: { decision, notes: req.body?.notes ?? null },
    ts: now()
  })
  c._decided = true
  res.json({ audit_id: id })
})

// SSE — backend bilan bir xil frame: "event: status\ndata: {...}\n\n"
app.get('/cases/:id/stream', (req, res) => {
  const c = cases.get(req.params.id)
  if (!c) return err(res, 404, 'not_found', 'Case topilmadi', { case_id: req.params.id })
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive'
  })
  res.flushHeaders?.()
  let last = null
  const tick = setInterval(() => {
    if (c.status !== last) {
      last = c.status
      res.write(`event: status\ndata: ${JSON.stringify({ case_id: c.case_id, status: c.status })}\n\n`)
    }
    if (c.status === 'DONE' || c.status === 'FAILED') {
      clearInterval(tick)
      res.end()
    }
  }, 300)
  req.on('close', () => clearInterval(tick))
})

app.listen(PORT, '127.0.0.1', () => {
  // eslint-disable-next-line no-console
  console.log(`[mock] §7 backend taqlidi: http://127.0.0.1:${PORT}`)
  console.log(`[mock] App Sozlamalar -> endpoint = http://127.0.0.1:${PORT}`)
})
