import { useEffect, useRef, useState } from 'react'
import type { CaseResult, CaseStatus, DecisionType } from '../../../shared/api-types'
import { getCase, pollCaseStatus, streamCase, submitDecision } from '../api/client'
import { ErrorBox } from '../components/ErrorBox'
import { ProgressTracker } from '../components/ProgressTracker'
import { RiskPanel } from '../components/RiskPanel'
import { DecisionBar } from '../components/DecisionBar'
import { ImageWithDetections } from '../components/ImageWithDetections'
import {
  AiBadge,
  AvailabilityBadge,
  DegradedBadge
} from '../components/ProvenanceBadge'

// Case natija ekrani: live progress -> §7.1 to'liq render -> operator qarori.
export function CaseView({
  caseId,
  imagePreviewUrl,
  onBack
}: {
  caseId: string
  imagePreviewUrl?: string | null
  onBack: () => void
}): JSX.Element {
  const [status, setStatus] = useState<CaseStatus>('PENDING')
  const [viaSse, setViaSse] = useState(true)
  const [result, setResult] = useState<CaseResult | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [decisionDone, setDecisionDone] = useState<string | null>(null) // audit_id
  const [submitting, setSubmitting] = useState(false)

  // --- Live progress: SSE, uzilsa polling fallback ---
  useEffect(() => {
    let stopPoll: (() => void) | null = null
    const handleStatus = (s: CaseStatus): void => setStatus(s)

    const stopSse = streamCase(caseId, {
      onStatus: (ev) => handleStatus(ev.status),
      onDone: () => void 0,
      onError: () => {
        // SSE uzildi -> polling'ga o'tamiz
        setViaSse(false)
        if (!stopPoll) stopPoll = pollCaseStatus(caseId, handleStatus)
      }
    })

    return () => {
      stopSse()
      stopPoll?.()
    }
  }, [caseId])

  // --- Terminal holatda to'liq natijani yuklash ---
  const loadedFor = useRef<string | null>(null)
  useEffect(() => {
    if ((status === 'DONE' || status === 'FAILED') && loadedFor.current !== status) {
      loadedFor.current = status
      getCase(caseId)
        .then(setResult)
        .catch(setError)
    }
  }, [status, caseId])

  const onDecision = async (decision: DecisionType, notes: string): Promise<void> => {
    setSubmitting(true)
    setError(null)
    try {
      const res = await submitDecision(caseId, { decision, notes: notes || null })
      setDecisionDone(res.audit_id)
    } catch (e) {
      setError(e)
    } finally {
      setSubmitting(false)
    }
  }

  const isHigh = result?.risk?.level === 'HIGH'

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 14 }}>
        <button className="btn ghost" type="button" onClick={onBack}>
          ← Orqaga
        </button>
        <span className="faint mono">case: {caseId}</span>
      </div>

      {error != null && <ErrorBox error={error} />}

      <ProgressTracker status={status} viaSse={viaSse} />

      {/* Natija hali yo'q (PENDING/PROCESSING) */}
      {status !== 'DONE' && status !== 'FAILED' && (
        <div className="card">
          <div className="row">
            <span className="spinner" />
            <span className="muted">Tahlil davom etmoqda. Natija tayyor bo&apos;lishi bilan ko&apos;rsatiladi…</span>
          </div>
        </div>
      )}

      {result && (
        <>
          {/* HIGH risk -> ko'zga tashlanadigan ALERT */}
          {isHigh && (
            <div className="high-alert">
              <span className="ico">🚨</span>
              <div>
                <strong>YUQORI XAVF (HIGH).</strong> Bu case operator e&apos;tiborini talab qiladi.
                Faktorlarni va rasmni diqqat bilan ko&apos;zdan kechiring.
              </div>
            </div>
          )}

          {/* degraded -> ochiq ko'rsat (qaysi qism ishlamadi) */}
          {result.degraded && (
            <div className="degraded-banner">
              <DegradedBadge />
              <div>
                Pipeline <b>qisman</b> ishladi — ba&apos;zi modellar (STT/LLM/detektsiya)
                bajarilmadi yoki timeout bo&apos;ldi. Quyidagi provenance belgilariga e&apos;tibor
                bering: yo&apos;q qism AI emas, balki <b>ma&apos;lumot yetishmasligi</b>.
              </div>
            </div>
          )}

          <div className="grid-2">
            {/* Chap: rasm + detektsiya overlay */}
            <div className="card">
              <div className="section-head">
                <h3>Skaner & aniqlangan obyektlar</h3>
                <span className="faint" style={{ fontSize: 11 }}>
                  {result.detections.length} ta obyekt
                </span>
              </div>
              {imagePreviewUrl ? (
                <ImageWithDetections src={imagePreviewUrl} detections={result.detections} />
              ) : (
                <div className="empty">Rasm preview bu sessiyada mavjud emas (backendda saqlangan).</div>
              )}
              {result.detections.length === 0 && (
                <p className="not-safe-note" style={{ marginTop: 12 }}>
                  Hech qanday obyekt aniqlanmadi — bu <b>xavfsizlik kafolati emas</b>. AI ko&apos;ra
                  olmagan narsa bo&apos;lishi mumkin.
                </p>
              )}
            </div>

            {/* O'ng: risk */}
            <RiskPanel risk={result.risk} />
          </div>

          {/* Tushuntirish — AI provenance bilan */}
          <div className="card">
            <div className="section-head">
              <h3>AI tushuntirishi</h3>
              <AiBadge
                by={result.explanation?.generated_by}
                available={!!result.explanation?.available}
              />
            </div>
            {result.explanation?.available && result.explanation.text ? (
              <p style={{ margin: 0 }}>{result.explanation.text}</p>
            ) : (
              <p className="muted" style={{ margin: 0 }}>
                AI tushuntirish bu case uchun mavjud emas. Bu <b>baho emas</b> — faqat tushuntirish
                generatsiya qilinmadi. Risk bahosi yuqorida deterministik hisoblangan.
              </p>
            )}
          </div>

          {/* Transcript — available bilan */}
          <div className="card">
            <div className="section-head">
              <h3>Ovoz transkripti</h3>
              <AvailabilityBadge available={!!result.transcript?.available} label="STT" />
            </div>
            {result.transcript?.available && result.transcript.text ? (
              <>
                <p style={{ margin: '0 0 8px' }}>{result.transcript.text}</p>
                <div className="kv">
                  <dt>Til</dt>
                  <dd>{result.transcript.language ?? '—'}</dd>
                  <dt>Ishonch</dt>
                  <dd>
                    {result.transcript.confidence != null
                      ? `${(result.transcript.confidence * 100).toFixed(0)}%`
                      : '—'}
                  </dd>
                </div>
              </>
            ) : (
              <p className="muted" style={{ margin: 0 }}>
                Audio berilmagan yoki transkripsiya mavjud emas.
              </p>
            )}
          </div>

          {/* Provenance xulosa + timings */}
          <div className="card">
            <h3>Manba & ishlash xulosasi (provenance)</h3>
            <p className="card-sub">Operator har qism qayerdan kelganini ko&apos;radi.</p>
            <div className="row wrap" style={{ gap: 8, marginBottom: 12 }}>
              <span className="prov det">⚙ Risk: {result.risk?.computed_by ?? '—'} (deterministik)</span>
              <AiBadge
                by={result.explanation?.generated_by}
                available={!!result.explanation?.available}
              />
              <AvailabilityBadge available={!!result.transcript?.available} label="Transkript" />
              {result.degraded && <DegradedBadge />}
            </div>
            {result.timings_ms && Object.keys(result.timings_ms).length > 0 && (
              <table className="factors">
                <thead>
                  <tr>
                    <th>Bosqich</th>
                    <th className="num">Vaqt (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.timings_ms).map(([k, v]) => (
                    <tr key={k}>
                      <td className="mono faint">{k}</td>
                      <td className="num">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Operator qarori */}
          {decisionDone ? (
            <div className="card">
              <h3>✓ Qaror yozildi</h3>
              <p className="muted">
                Audit yozuvi yaratildi: <span className="mono">{decisionDone}</span>
              </p>
              <button className="btn secondary" type="button" onClick={onBack}>
                Yangi case
              </button>
            </div>
          ) : (
            <DecisionBar
              disabled={status === 'FAILED'}
              submitting={submitting}
              onSubmit={(d, n) => void onDecision(d, n)}
            />
          )}
        </>
      )}

      <p className="assistant-note">
        AI yordamchi signal berdi — yakuniy <b>qaror operatorники</b>. Past risk ham xavfsizlik
        kafolati emas.
      </p>
    </div>
  )
}
