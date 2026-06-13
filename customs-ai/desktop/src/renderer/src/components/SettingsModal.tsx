import { useState } from 'react'
import { DEFAULT_BASE_URL, getBaseUrl, setBaseUrl } from '../api/config'

// Endpoint sozlash (ADR-005, Faza 2: 127.0.0.1 -> server IP).
export function SettingsModal({ onClose }: { onClose: () => void }): JSX.Element {
  const [url, setUrl] = useState(getBaseUrl())

  const save = (): void => {
    setBaseUrl(url)
    onClose()
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2 style={{ marginTop: 0 }}>Sozlamalar</h2>
        <p className="card-sub">
          Backend endpoint. Offline tizim — manzil 127.0.0.1 (lokal) yoki LAN server IP bo&apos;lishi
          mumkin (Faza 2). Faqat HTTP/REST (ADR-005).
        </p>
        <label className="field">
          <span className="lbl">Backend manzili (base URL)</span>
          <input
            type="url"
            value={url}
            spellCheck={false}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={DEFAULT_BASE_URL}
          />
        </label>
        <div className="row" style={{ justifyContent: 'space-between', marginTop: 8 }}>
          <button className="btn ghost" type="button" onClick={() => setUrl(DEFAULT_BASE_URL)}>
            Standart ({DEFAULT_BASE_URL})
          </button>
          <div className="row">
            <button className="btn secondary" type="button" onClick={onClose}>
              Bekor
            </button>
            <button className="btn" type="button" onClick={save}>
              Saqlash
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
