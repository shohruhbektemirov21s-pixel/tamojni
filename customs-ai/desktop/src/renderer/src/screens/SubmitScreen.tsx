import { useRef, useState } from 'react'
import { createCase } from '../api/client'
import { ErrorBox } from '../components/ErrorBox'

// Case yuborish: X-ray rasm (MAJBURIY) + audio (ixtiyoriy) + matn izoh (ixtiyoriy).
export function SubmitScreen({
  onCreated
}: {
  // case yaratilgach, id va lokal rasm preview URL'ini yuqoriga uzatamiz.
  onCreated: (caseId: string, imagePreviewUrl: string) => void
}): JSX.Element {
  const [image, setImage] = useState<File | null>(null)
  const [audio, setAudio] = useState<File | null>(null)
  const [notes, setNotes] = useState('')
  const [operatorId, setOperatorId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [dragImg, setDragImg] = useState(false)

  const imgInput = useRef<HTMLInputElement>(null)
  const audInput = useRef<HTMLInputElement>(null)

  const submit = async (): Promise<void> => {
    if (!image) return
    setBusy(true)
    setError(null)
    try {
      const res = await createCase({
        image,
        audio,
        notes: notes.trim() || null,
        operatorId: operatorId.trim() || null
      })
      onCreated(res.case_id, URL.createObjectURL(image))
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="container" style={{ maxWidth: 760 }}>
      <div className="card">
        <h2>Yangi case yuborish</h2>
        <p className="card-sub">
          X-ray skanini yuklang. Ovoz va izoh ixtiyoriy. Tizim tahlil qiladi — <b>qaror sizniki</b>.
        </p>

        {error != null && <ErrorBox error={error} onRetry={() => setError(null)} />}

        {/* X-ray — majburiy */}
        <label className="field">
          <span className="lbl">
            X-ray rasm<span className="req">* majburiy</span>
          </span>
          <div
            className={`dropzone ${dragImg ? 'drag' : ''} ${image ? 'has-file' : ''}`}
            onClick={() => imgInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragImg(true)
            }}
            onDragLeave={() => setDragImg(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragImg(false)
              const f = e.dataTransfer.files?.[0]
              if (f && f.type.startsWith('image/')) setImage(f)
            }}
          >
            {image ? (
              <>
                ✓ <b>{image.name}</b>{' '}
                <span className="faint">({(image.size / 1024).toFixed(0)} KB)</span>
                <div className="faint" style={{ fontSize: 12 }}>
                  almashtirish uchun bosing
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 24 }}>🖼</div>
                Rasmni shu yerga tashlang yoki <b>tanlash uchun bosing</b>
              </>
            )}
          </div>
          <input
            ref={imgInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => setImage(e.target.files?.[0] ?? null)}
          />
        </label>

        {/* Audio — ixtiyoriy */}
        <label className="field">
          <span className="lbl">
            Ovozli izoh<span className="opt">(ixtiyoriy)</span>
          </span>
          <div className="row">
            <button className="btn secondary" type="button" onClick={() => audInput.current?.click()}>
              {audio ? 'Audioni almashtirish' : 'Audio tanlash'}
            </button>
            {audio && (
              <span className="muted">
                🎙 {audio.name}{' '}
                <button
                  className="btn ghost"
                  type="button"
                  style={{ padding: '2px 8px', marginLeft: 6 }}
                  onClick={() => setAudio(null)}
                >
                  o&apos;chirish
                </button>
              </span>
            )}
          </div>
          <input
            ref={audInput}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => setAudio(e.target.files?.[0] ?? null)}
          />
        </label>

        {/* Matn izoh — ixtiyoriy */}
        <label className="field">
          <span className="lbl">
            Matn izoh<span className="opt">(ixtiyoriy)</span>
          </span>
          <textarea
            value={notes}
            placeholder="Operatorning kuzatuvi yoki konteksti..."
            onChange={(e) => setNotes(e.target.value)}
          />
        </label>

        <label className="field">
          <span className="lbl">
            Operator ID<span className="opt">(ixtiyoriy, audit uchun)</span>
          </span>
          <input
            type="text"
            value={operatorId}
            placeholder="masalan: op-1707"
            onChange={(e) => setOperatorId(e.target.value)}
          />
        </label>

        <button className="btn" disabled={!image || busy} onClick={submit} type="button">
          {busy ? 'Yuborilmoqda…' : 'Tahlilga yuborish'}
        </button>
        {!image && <span className="faint" style={{ marginLeft: 12 }}>X-ray rasm majburiy</span>}
      </div>

      <p className="assistant-note">
        Tizim AI yordamchidir, hakam emas. U signal va tushuntirish beradi — yakuniy qarorni operator
        qabul qiladi.
      </p>
    </div>
  )
}
