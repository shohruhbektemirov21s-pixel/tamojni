// HTTP klient — faqat REST/SSE (ADR-005). Backend xatoligini operator tushunadigan
// {error:{code,message,detail}} formatida ApiError'ga aylantiradi.
import type {
  ApiErrorBody,
  AuditList,
  CaseCreated,
  CaseList,
  CaseResult,
  DecisionInput,
  DecisionResult,
  Health,
  RiskLevel,
  CaseStatus,
  StreamStatusEvent
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

// ---------------- SSE: progress (polling fallback bilan) ----------------

export interface StreamHandlers {
  onStatus: (ev: StreamStatusEvent) => void
  onError?: (err: ApiError) => void
  onDone?: () => void
}

const TERMINAL: CaseStatus[] = ['DONE', 'FAILED']

/**
 * GET /cases/{id}/stream (SSE). EventSource backend endpoint'iga ulanadi.
 * Ulanish uzilsa, chaqiruvchi polling fallback'ga o'tishi mumkin (qaytgan stop()
 * orqali). Terminal holatda avtomatik yopiladi.
 */
export function streamCase(caseId: string, h: StreamHandlers): () => void {
  const url = `${getBaseUrl()}/cases/${encodeURIComponent(caseId)}/stream`
  const es = new EventSource(url)
  let closed = false

  const close = (): void => {
    if (closed) return
    closed = true
    es.close()
  }

  es.addEventListener('status', (ev) => {
    try {
      const data = JSON.parse((ev as MessageEvent).data) as StreamStatusEvent
      h.onStatus(data)
      if (TERMINAL.includes(data.status)) {
        close()
        h.onDone?.()
      }
    } catch {
      /* buzuq frame — e'tiborsiz */
    }
  })

  es.onerror = () => {
    // SSE uzildi: chaqiruvchi polling'ga o'tadi.
    if (!closed) h.onError?.(new ApiError('network_error', 'SSE ulanishi uzildi', 0))
  }

  return close
}

/** SSE ishlamasa polling fallback: terminal holatga yetguncha so'raydi. */
export function pollCaseStatus(
  caseId: string,
  onStatus: (s: CaseStatus) => void,
  intervalMs = 1000
): () => void {
  let stopped = false
  let last: CaseStatus | null = null
  const tick = async (): Promise<void> => {
    if (stopped) return
    try {
      const c = await getCase(caseId)
      if (c.status !== last) {
        last = c.status
        onStatus(c.status)
      }
      if (TERMINAL.includes(c.status)) return
    } catch {
      /* keyingi tick'da qayta urinadi */
    }
    if (!stopped) setTimeout(tick, intervalMs)
  }
  void tick()
  return () => {
    stopped = true
  }
}
