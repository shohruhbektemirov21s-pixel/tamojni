import { useState } from 'react'
import { SubmitScreen } from './screens/SubmitScreen'
import { CaseView } from './screens/CaseView'
import { HealthBadge } from './components/HealthBadge'
import { SettingsModal } from './components/SettingsModal'

type View = { name: 'submit' } | { name: 'case'; caseId: string; imageUrl: string }

export default function App(): JSX.Element {
  const [view, setView] = useState<View>({ name: 'submit' })
  const [settings, setSettings] = useState(false)

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          🛂 Bojxona Operator
          <small>Offline AI yordamchi</small>
        </span>
        <div className="spacer" />
        <HealthBadge />
        <button className="btn ghost" type="button" onClick={() => setSettings(true)}>
          ⚙ Sozlamalar
        </button>
      </header>

      <main className="content">
        {view.name === 'submit' && (
          <SubmitScreen
            onCreated={(caseId, imageUrl) => setView({ name: 'case', caseId, imageUrl })}
          />
        )}
        {view.name === 'case' && (
          <CaseView
            caseId={view.caseId}
            imagePreviewUrl={view.imageUrl}
            onBack={() => setView({ name: 'submit' })}
          />
        )}
      </main>

      {settings && <SettingsModal onClose={() => setSettings(false)} />}
    </div>
  )
}
