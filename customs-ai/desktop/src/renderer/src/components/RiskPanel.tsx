import type { Risk } from '../../../shared/api-types'
import { DeterministicBadge } from './ProvenanceBadge'

// Risk darajasi + score + FAKTORLAR ("nega shu daraja"). Risk DETERMINISTIK
// (computed_by) — buni ochiq belgilaymiz, AI bilan aralashtirmaymiz (Tamoyil 2).
export function RiskPanel({ risk }: { risk: Risk | null }): JSX.Element {
  if (!risk) {
    return (
      <div className="card">
        <h3>Risk bahosi</h3>
        <p className="muted">Risk hisoblanmadi (case tugamagan yoki ma&apos;lumot yetarli emas).</p>
      </div>
    )
  }

  const maxContrib = Math.max(0.0001, ...risk.factors.map((f) => Number(f.contribution) || 0))

  return (
    <div className="card">
      <div className="section-head">
        <h3>Risk bahosi</h3>
        <DeterministicBadge by={risk.computed_by} />
      </div>

      <div className={`risk-banner risk-${risk.level}`}>
        <div>
          <div className="level">{risk.level}</div>
          <div className="score">
            Score: <span className="mono">{risk.score.toFixed(3)}</span>
          </div>
        </div>
      </div>

      <div className="section-head" style={{ marginTop: 16 }}>
        <h3 style={{ fontSize: 13 }}>Nega shu daraja? — faktorlar</h3>
      </div>

      {risk.factors.length === 0 ? (
        <p className="muted">Hech qanday xavf signali topilmadi.</p>
      ) : (
        <table className="factors">
          <thead>
            <tr>
              <th>Qoida</th>
              <th>Sinf</th>
              <th className="num">Ishonch</th>
              <th className="num">Og&apos;irlik</th>
              <th className="num">Hissa</th>
              <th style={{ width: 120 }}></th>
            </tr>
          </thead>
          <tbody>
            {risk.factors.map((f, i) => {
              const contrib = Number(f.contribution) || 0
              return (
                <tr key={i}>
                  <td className="mono faint">{f.rule}</td>
                  <td>{f.class ?? '—'}</td>
                  <td className="num">
                    {f.confidence != null ? `${(Number(f.confidence) * 100).toFixed(0)}%` : '—'}
                  </td>
                  <td className="num">{f.weight != null ? Number(f.weight).toFixed(2) : '—'}</td>
                  <td className="num">{contrib.toFixed(3)}</td>
                  <td>
                    <div className="bar" style={{ width: `${(contrib / maxContrib) * 100}%` }} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}

      {/* Tamoyil 3: "Topilmadi != xavfsiz" — LOW bo'lsa ham ortiqcha tinchlantirma */}
      {risk.level === 'LOW' && (
        <div className="not-safe-note">
          ⚠ <strong>Past risk — lekin kafolat emas.</strong> AI ko&apos;rmagan yoki
          o&apos;rganmagan tahdid bo&apos;lishi mumkin. Yakuniy qaror operatorning ko&apos;zdan
          kechirishiga asoslanadi.
        </div>
      )}
    </div>
  )
}
