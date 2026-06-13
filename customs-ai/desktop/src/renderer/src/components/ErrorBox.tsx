import { ApiError } from '../api/client'

// Backend xatoligini operator tushunadigan tilda ko'rsatadi (code + message + detail).
export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }): JSX.Element {
  const isApi = error instanceof ApiError
  const msg = isApi ? error.operatorMessage : String((error as Error)?.message ?? error)
  const code = isApi ? error.code : 'error'
  const detail = isApi ? error.detail : undefined

  return (
    <div className="error-box">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <strong>Xatolik</strong>
        <span className="code">{code}</span>
      </div>
      <div style={{ marginTop: 4 }}>{msg}</div>
      {detail != null && (
        <pre className="mono faint" style={{ fontSize: 11, marginTop: 6, whiteSpace: 'pre-wrap' }}>
          {typeof detail === 'string' ? detail : JSON.stringify(detail)}
        </pre>
      )}
      {onRetry && (
        <button className="btn secondary" style={{ marginTop: 8 }} onClick={onRetry} type="button">
          Qayta urinish
        </button>
      )}
    </div>
  )
}
