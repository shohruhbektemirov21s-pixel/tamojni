// HTTP klient (REST) + WebSocket real-time push (ADR-005). Backend xatoligini
// operator tushunadigan {error:{code,message,detail}} formatida ApiError'ga aylantiradi.
import type {
  ApiErrorBody,
  AuditList,
  CaseCreated,
  CaseList,
  CaseResult,
  DecisionInput,
  DecisionResult,
  Detection,
  Explanation,
  Health,
  Risk,
  RiskLevel,
  Transcript,
  CaseStatus,
  WsEvent
} from '../../../shared/api-types'
import { getBaseUrl } from './config'

/** Backend xatoligi (kontrakt envelope'i) yoki tarmoq/timeout xatoligi. */
export class ApiError extends Error {
  readonly code: string
  readonly httpStatus: number
  readonly detail: unknown

  constructor(code: string, message: string, httpStatus: number, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.httpStatus = httpStatus
    this.detail = detail
  }

  /** Operatorga ko'rsatish uchun qisqa, tushunarli matn. */
  get operatorMessage(): string {
    switch (this.code) {
      case 'network_error':
        return "Backend bilan aloqa yo'q. Xizmat ishlayotganini va endpoint to'g'riligini tekshiring."
      case 'timeout':
        return 'Backend javob bermadi (timeout). Qayta urinib ko\'ring.'
      case 'validation_failed':
        return `So'rov noto'g'ri: ${this.message}`
      case 'disk_full':
      case 'service_unavailable':
        return `Xizmat vaqtincha mavjud emas: ${this.message}`
      case 'not_found':
        return `Topilmadi: ${this.message}`
      default:
        return this.message
    }
  }
}

const DEFAULT_TIMEOUT_MS = 30_000

async function parseError(res: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> | null = null
  try {
    body = (await res.json()) as ApiErrorBody
  } catch {
    /* JSON emas — quyida fallback */
  }
  const e = body?.error
  if (e?.code) {
    return new ApiError(e.code, e.message ?? `HTTP ${res.status}`, res.status, e.detail)
  }
  return new ApiError('http_error', `Kutilmagan xatolik (HTTP ${res.status})`, res.status)
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  let res: Response
  try {
    res = await fetch(`${getBaseUrl()}${path}`, { ...init, signal: ctrl.signal })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('timeout', 'So\'rov vaqti tugadi', 0)
    }
    throw new ApiError('network_error', String((err as Error)?.message ?? err), 0)
  } finally {
    clearTimeout(timer)
  }

  if (!res.ok) throw await parseError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

// ---------------- §7 endpointlar ----------------

export function getHealth(): Promise<Health> {
  return request<Health>('/health', {}, 5_000)
}

export interface CreateCaseInput {
  image: File // majburiy (X-ray)
  audio?: File | null // ixtiyoriy
  notes?: string | null // ixtiyoriy
  operatorId?: string | null
}

export function createCase(input: CreateCaseInput): Promise<CaseCreated> {
  const fd = new FormData()
  fd.append('image', input.image, input.image.name)
  if (input.audio) fd.append('audio', input.audio, input.audio.name)
  if (input.notes) fd.append('notes', input.notes)
  if (input.operatorId) fd.append('operator_id', input.operatorId)
  // multipart bo'lgani uchun Content-Type'ni qo'lda qo'ymaymiz (boundary auto).
  return request<CaseCreated>('/cases', { method: 'POST', body: fd }, 60_000)
}

export function getCase(caseId: string): Promise<CaseResult> {
  return request<CaseResult>(`/cases/${encodeURIComponent(caseId)}`)
}

export function getAudit(caseId: string): Promise<AuditList> {
  return request<AuditList>(`/cases/${encodeURIComponent(caseId)}/audit`)
}

export interface ListCasesQuery {
  status?: CaseStatus
  risk_level?: RiskLevel
  limit?: number
  offset?: number
}

export function listCases(q: ListCasesQuery = {}): Promise<CaseList> {
  const params = new URLSearchParams()
  if (q.status) params.set('status', q.status)
  if (q.risk_level) params.set('risk_level', q.risk_level)
  if (q.limit != null) params.set('limit', String(q.limit))
  if (q.offset != null) params.set('offset', String(q.offset))
  const qs = params.toString()
  return request<CaseList>(`/cases${qs ? `?${qs}` : ''}`)
}

export function submitDecision(caseId: string, input: DecisionInput): Promise<DecisionResult> {
  return request<DecisionResult>(`/cases/${encodeURIComponent(caseId)}/decision`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input)
  })
}

// ---------------- WebSocket /ws: real-time push (Polling YO'Q) ----------------

/** REST base'dan WS URL'ini olamiz: http->ws, https->wss. */
function wsUrl(path: string): string {
  const base = getBaseUrl()
  const scheme = base.replace(/^http:/i, 'ws:').replace(/^https:/i, 'wss:')
  return `${scheme}${path}`
}

/**
 * Jonli pipeline eventlari — har biri ixtiyoriy; UI faqat keragini ulaydi.
 * Backend `data` shakllari worker.py/events.py dan AYNAN olingan.
 */
export interface CaseEventHandlers {
  onStatus?: (s: CaseStatus) => void
  /** Tier 1 (<1s): detect + DETERMINISTIK risk darhol keladi. */
  onTier1?: (detections: Detection[], risk: Pick<Risk, 'level' | 'score' | 'computed_by'>) => void
  /** HIGH risk -> operatorga darhol signal (terminal'ni kutmaydi). */
  onAlert?: (level: RiskLevel) => void
  /** Jonli transkript bo'lagi (typing). */
  onSttPartial?: (text: string, isFinal: boolean) => void
  onSttDone?: (t: Pick<Transcript, 'text' | 'language' | 'available'>) => void
  /** Jonli LLM token — qabul qilingan tartibda biriktiriladi. */
  onExplanationToken?: (token: string) => void
  onExplanationDone?: (e: Pick<Explanation, 'text' | 'generated_by' | 'available'>) => void
  onTtsReady?: (meta: Record<string, unknown>) => void
  onDegraded?: (stage: string) => void
  /** WS ulandi/qayta ulandi. */
  onOpen?: () => void
  /** WS uzildi (avto qayta ulanish boshlandi). */
  onConnLost?: () => void
}

/**
 * `/ws?case_id=...` WebSocket'iga ulanadi va pipeline push eventlarini handler'larga
 * yo'naltiradi. Polling YO'Q — uzilsa eksponensial backoff bilan avto qayta ulanadi.
 * Qaytgan funksiya ulanishni butunlay yopadi (komponent unmount'da).
 */
export function connectEvents(caseId: string, h: CaseEventHandlers): () => void {
  let closed = false
  let ws: WebSocket | null = null
  let pingTimer: ReturnType<typeof setInterval> | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let backoff = 500

  const clearPing = (): void => {
    if (pingTimer) {
      clearInterval(pingTimer)
      pingTimer = null
    }
  }

  const dispatch = (ev: WsEvent): void => {
    const d = ev.data as Record<string, unknown>
    switch (ev.type) {
      case 'ready':
        break
      case 'scan_ingested':
      case 'case_created':
        h.onStatus?.('PROCESSING')
        break
      case 'tier1_done':
        h.onStatus?.('PROCESSING')
        if (d.risk) {
          h.onTier1?.(
            (d.detections as Detection[]) ?? [],
            d.risk as Pick<Risk, 'level' | 'score' | 'computed_by'>
          )
        }
        break
      case 'alert':
        h.onAlert?.((d.level as RiskLevel) ?? 'HIGH')
        break
      case 'stt_partial':
        h.onSttPartial?.(String(d.text ?? ''), Boolean(d.is_final))
        break
      case 'stt_done':
        h.onSttDone?.({
          text: (d.text as string | null) ?? null,
          language: (d.language as string | null) ?? null,
          available: Boolean(d.available)
        })
        break
      case 'explanation_token':
        h.onExplanationToken?.(String(d.token ?? ''))
        break
      case 'explanation_done':
        h.onExplanationDone?.({
          text: (d.text as string | null) ?? null,
          generated_by: (d.generated_by as string | null) ?? null,
          available: Boolean(d.available)
        })
        break
      case 'tts_ready':
        h.onTtsReady?.(d)
        break
      case 'model_failed':
        h.onDegraded?.(String(d.stage ?? 'unknown'))
        break
      case 'case_done':
        h.onStatus?.('DONE')
        break
      case 'case_failed':
        h.onStatus?.('FAILED')
        break
      default:
        break
    }
  }

  const scheduleReconnect = (): void => {
    if (closed || reconnectTimer) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, backoff)
    backoff = Math.min(backoff * 2, 5_000)
  }

  const connect = (): void => {
    if (closed) return
    try {
      ws = new WebSocket(wsUrl(`/ws?case_id=${encodeURIComponent(caseId)}`))
    } catch {
      scheduleReconnect()
      return
    }
    ws.onopen = (): void => {
      backoff = 500
      h.onOpen?.()
      // keep-alive: backend recv loop uzilishni shu orqali ham aniqlaydi
      pingTimer = setInterval(() => {
        try {
          if (ws?.readyState === WebSocket.OPEN) ws.send('ping')
        } catch {
          /* yopilayotgan bo'lishi mumkin */
        }
      }, 25_000)
    }
    ws.onmessage = (m: MessageEvent): void => {
      try {
        dispatch(JSON.parse(String(m.data)) as WsEvent)
      } catch {
        /* buzuq frame — e'tiborsiz */
      }
    }
    ws.onerror = (): void => {
      /* onclose qayta ulanishni boshqaradi */
    }
    ws.onclose = (): void => {
      clearPing()
      if (closed) return
      h.onConnLost?.()
      scheduleReconnect()
    }
  }

  connect()

  return (): void => {
    closed = true
    clearPing()
    if (reconnectTimer) clearTimeout(reconnectTimer)
    try {
      ws?.close()
    } catch {
      /* noop */
    }
  }
}
