import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'

// Backend mavjudligi indikatori. Offline ilova bo'lgani uchun operator
// xizmat ishlayotganini bir qarashda ko'rishi muhim.
type State = 'checking' | 'ok' | 'down'

export function HealthBadge(): JSX.Element {
  const [state, setState] = useState<State>('checking')
  const [models, setModels] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    let alive = true
    const check = async (): Promise<void> => {
      try {
        const h = await getHealth()
        if (!alive) return
        setState('ok')
        setModels(h.models ?? null)
      } catch {
        if (!alive) return
        setState('down')
        setModels(null)
      }
    }
    void check()
    const t = setInterval(check, 8000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  const label =
    state === 'checking' ? 'Tekshirilmoqda…' : state === 'ok' ? 'Backend ulangan' : 'Backend uzilgan'
  const dotcls = state === 'ok' ? 'ok' : state === 'down' ? 'bad' : 'warn'
  const title = models ? `Modellar: ${JSON.stringify(models)}` : 'Modellar holati noma\'lum'

  return (
    <span className="health" title={title}>
      <span className={`dot ${dotcls}`} />
      {label}
    </span>
  )
}
