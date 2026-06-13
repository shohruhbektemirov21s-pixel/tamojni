import type { CaseStatus } from '../../../shared/api-types'

// Live progress: PENDING -> PROCESSING -> DONE (yoki FAILED).
const ORDER: CaseStatus[] = ['PENDING', 'PROCESSING', 'DONE']
const LABEL: Record<CaseStatus, string> = {
  PENDING: 'Navbatda',
  PROCESSING: 'Tahlil qilinmoqda',
  DONE: 'Tayyor',
  FAILED: 'Xatolik'
}

export function ProgressTracker({
  status,
  live
}: {
  status: CaseStatus
  live: boolean
}): JSX.Element {
  const failed = status === 'FAILED'
  const idx = ORDER.indexOf(status === 'FAILED' ? 'PROCESSING' : status)

  return (
    <div className="card">
      <div className="section-head">
        <h3>Holat</h3>
        <span className="faint" style={{ fontSize: 11 }}>
          {live ? '● jonli (WebSocket)' : '○ ulanmoqda…'}
        </span>
      </div>
      <div className="stepper">
        {ORDER.map((s, i) => {
          const state = failed && i === ORDER.length - 1 ? 'failed' : i < idx ? 'done' : i === idx ? 'active' : ''
          return (
            <div key={s} style={{ display: 'contents' }}>
              <div className={`step ${state}`}>
                <span className="num">
                  {state === 'done' ? '✓' : state === 'failed' ? '✕' : i + 1}
                </span>
                <span>{LABEL[s]}</span>
                {state === 'active' && !failed && <span className="spinner" />}
              </div>
              {i < ORDER.length - 1 && <div className={`step-line ${i < idx ? 'done' : ''}`} />}
            </div>
          )
        })}
      </div>
      {failed && (
        <div className="error-box" style={{ marginTop: 12, marginBottom: 0 }}>
          Tahlil bajarilmadi (FAILED). Case audit jurnalida sababni ko&apos;ring yoki qayta
          yuboring.
        </div>
      )}
    </div>
  )
}
