// §7 / §7.1 API kontrakti — backend serializers.py va schemas.py dan AYNAN olingan.
// Bu tiplar real backend va mock-server o'rtasida bir xil (yagona haqiqat manbai).

export type CaseStatus = 'PENDING' | 'PROCESSING' | 'DONE' | 'FAILED'
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH'
export type DecisionType = 'CONFIRM' | 'REJECT' | 'OVERRIDE'

// --- §7 xatolik envelope'i: {"error":{"code","message","detail"}} ---
export interface ApiErrorBody {
  error: {
    code: string // validation_failed | not_found | conflict | disk_full | service_unavailable | ...
    message: string
    detail?: unknown
  }
}

// --- GET /health ---
export interface Health {
  status: string
  models: Record<string, unknown> // {yolo, stt, llm}
  gpu: Record<string, unknown>
}

// --- §7.1 Case Result obyekti ---
export interface Detection {
  class: string
  confidence: number // 0..1
  bbox: [number, number, number, number] // [x1,y1,x2,y2] ASL rasm pikselida
}

// risk.factors[] — "nega shu daraja": har bir hissa qo'shgan qoida (audit-ready)
export interface RiskFactor {
  rule: string // masalan "prohibited_class"
  class?: string
  confidence?: number
  weight?: number
  contribution?: number
  [k: string]: unknown // boshqa qoidalar boshqa maydonlar berishi mumkin
}

export interface Risk {
  level: RiskLevel
  score: number
  computed_by: string // DETERMINISTIK manbai, masalan "rule_engine_ref_v1"
  factors: RiskFactor[]
}

export interface Transcript {
  text: string | null
  language: string | null
  confidence: number | null
  available: boolean
}

export interface Explanation {
  text: string | null
  generated_by: string | null // AI model versiyasi (provenance)
  available: boolean
}

export interface CaseResult {
  case_id: string
  status: CaseStatus
  risk: Risk | null
  detections: Detection[]
  transcript: Transcript | null
  explanation: Explanation | null
  operator_notes: string | null
  timings_ms: Record<string, number> | null
  degraded: boolean
}

// --- POST /cases -> 201 ---
export interface CaseCreated {
  case_id: string
  status: CaseStatus // "PENDING"
}

// --- GET /cases (ro'yxat) ---
export interface CaseListItem {
  case_id: string
  status: CaseStatus
  risk_level: RiskLevel | null
  degraded: boolean
  created_at: string // ISO
}

export interface CaseList {
  items: CaseListItem[]
  total: number
}

// --- GET /cases/{id}/audit ---
export interface AuditEntry {
  id: string
  actor: string
  action: string
  payload: Record<string, unknown> | null
  ts: string // ISO
}

export interface AuditList {
  entries: AuditEntry[]
}

// --- POST /cases/{id}/decision ---
export interface DecisionInput {
  decision: DecisionType
  notes?: string | null
  operator_id?: string | null
}

export interface DecisionResult {
  audit_id: string
}

// --- SSE: GET /cases/{id}/stream -> event: status, data: {case_id, status} ---
// (legacy coarse transport — real-time oqim endi WebSocket /ws orqali, pastda)
export interface StreamStatusEvent {
  case_id: string
  status: CaseStatus
}

// --- WebSocket /ws push konverti (§7 schema) — ASOSIY real-time transport ---
// Backend events.py EventType bilan AYNAN bir xil (yagona haqiqat manbai).
// Polling YO'Q: pipeline har bosqichni shu yerga push qiladi, UI darhol render qiladi.
export type WsEventType =
  | 'ready' // ulanish tasdig'i (server -> client, seq=0)
  | 'scan_ingested'
  | 'scan_rejected'
  | 'case_created'
  | 'tier1_done' // detect + DETERMINISTIK risk (<1s)
  | 'alert' // HIGH risk -> darhol signal
  | 'stt_partial' // jonli transkript bo'lagi
  | 'stt_done'
  | 'explanation_token' // jonli LLM token (typing effekti)
  | 'explanation_done'
  | 'tts_ready'
  | 'case_done'
  | 'case_failed'
  | 'model_failed' // degradatsiya (Tamoyil 6)
  | 'backpressure'

export interface WsEvent<T = Record<string, unknown>> {
  type: WsEventType
  case_id: string | null
  seq: number // bus bo'yicha monotonik — tartib/qayta-ulanish uchun
  ts: number
  data: T
}
