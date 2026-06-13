import { useState } from 'react'
import type { DecisionType } from '../../../shared/api-types'

// Operator qarori: CONFIRM / REJECT / OVERRIDE + izoh -> POST /cases/{id}/decision.
// QAROR OPERATORDA (Tamoyil 1) — AI faqat signal berdi.
const OPTIONS: { type: DecisionType; label: string; hint: string; cls: string }[] = [
  { type: 'CONFIRM', label: 'TASDIQLASH', hint: 'AI bahosiga qo\'shilaman', cls: 'confirm' },
  { type: 'REJECT', label: 'RAD ETISH', hint: 'Xavf yo\'q / xato signal', cls: 'reject' },
  { type: 'OVERRIDE', label: 'BEKOR QILISH', hint: 'AI dan farqli qaror', cls: 'override' }
]

export function DecisionBar({
  disabled,
  submitting,
  onSubmit
}: {
  disabled?: boolean
  submitting?: boolean
  onSubmit: (decision: DecisionType, notes: string) => void
}): JSX.Element {
  const [sel, setSel] = useState<DecisionType | null>(null)
  const [notes, setNotes] = useState('')

  // OVERRIDE va REJECT uchun izoh majburiy (audit uchun sabab kerak).
  const notesRequired = sel === 'OVERRIDE' || sel === 'REJECT'
  const canSubmit = !!sel && !disabled && !submitting && (!notesRequired || notes.trim().length > 0)

  return (
    <div className="card">
      <div className="section-head">
        <h3>Operator qarori</h3>
        <span className="faint" style={{ fontSize: 11 }}>
          Yakuniy qaror — sizniki. AI faqat yordamchi.
        </span>
      </div>

      <div className="decision-bar">
        {OPTIONS.map((o) => (
          <button
            key={o.type}
            className={`dbtn ${o.cls} ${sel === o.type ? 'sel' : ''}`}
            disabled={disabled || submitting}
            onClick={() => setSel(o.type)}
            type="button"
          >
            {o.label}
            <small>{o.hint}</small>
          </button>
        ))}
      </div>

      <label className="field" style={{ marginTop: 14 }}>
        <span className="lbl">
          Izoh
          {notesRequired ? <span className="req">* (majburiy)</span> : <span className="opt">(ixtiyoriy)</span>}
        </span>
        <textarea
          value={notes}
          placeholder={
            notesRequired
              ? 'Qaroringiz sababini yozing (audit uchun)...'
              : 'Qo\'shimcha izoh (ixtiyoriy)...'
          }
          onChange={(e) => setNotes(e.target.value)}
          disabled={disabled || submitting}
        />
      </label>

      <button
        className="btn"
        disabled={!canSubmit}
        onClick={() => sel && onSubmit(sel, notes.trim())}
        type="button"
      >
        {submitting ? 'Yozilmoqda…' : 'Qarorni saqlash (audit)'}
      </button>
    </div>
  )
}
