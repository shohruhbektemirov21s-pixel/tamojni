// Provenance belgisi — operator qaysi qism AI'dan, qaysi qism DETERMINISTIK
// ekanini, mavjud/yo'qligini bir qarashda ko'radi (Tamoyil 2: shaffoflik).

export function AiBadge({ by, available }: { by?: string | null; available: boolean }): JSX.Element {
  if (!available) {
    return (
      <span className="prov na" title="AI tushuntirish bu case uchun mavjud emas">
        ⚠ AI mavjud emas
      </span>
    )
  }
  return (
    <span className="prov ai" title={`AI tomonidan generatsiya qilingan${by ? ` — ${by}` : ''}`}>
      🧠 AI{by ? ` · ${by}` : ''}
    </span>
  )
}

export function DeterministicBadge({ by }: { by?: string | null }): JSX.Element {
  return (
    <span
      className="prov det"
      title={`Deterministik qoidalar bilan hisoblangan${by ? ` — ${by}` : ''} (AI emas)`}
    >
      ⚙ Deterministik{by ? ` · ${by}` : ''}
    </span>
  )
}

export function AvailabilityBadge({
  available,
  label
}: {
  available: boolean
  label: string
}): JSX.Element {
  return available ? (
    <span className="prov det" title={`${label} mavjud`}>
      ✓ {label}
    </span>
  ) : (
    <span className="prov na" title={`${label} mavjud emas`}>
      — {label} yo&apos;q
    </span>
  )
}

export function DegradedBadge(): JSX.Element {
  return (
    <span
      className="prov degraded"
      title="Pipeline qisman ishladi — ba'zi modellar ishlamadi yoki timeout bo'ldi"
    >
      ⚠ Degraded
    </span>
  )
}
