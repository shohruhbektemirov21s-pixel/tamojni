// Mock /ws jonli oqim smoke-testi: case yaratadi, WS'ga ulanadi, eventlarni sanaydi.
import net from 'net'
import crypto from 'crypto'

const PORT = process.env.MOCK_PORT ? Number(process.env.MOCK_PORT) : 8787
const BASE = `http://127.0.0.1:${PORT}`

function decodeFrames(buf, onText) {
  let off = 0
  while (off + 2 <= buf.length) {
    const b1 = buf[off + 1]
    let len = b1 & 0x7f
    let headerLen = 2
    if (len === 126) {
      if (off + 4 > buf.length) break
      len = buf.readUInt16BE(off + 2)
      headerLen = 4
    } else if (len === 127) {
      if (off + 10 > buf.length) break
      len = Number(buf.readBigUInt64BE(off + 2))
      headerLen = 10
    }
    if (off + headerLen + len > buf.length) break
    const payload = buf.slice(off + headerLen, off + headerLen + len)
    onText(payload.toString('utf8'))
    off += headerLen + len
  }
  return buf.slice(off)
}

async function main() {
  // 1) case yaratamiz (multipart)
  const fd = new FormData()
  fd.append('image', new Blob([Buffer.from([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4])]), 'x.png')
  fd.append('audio', new Blob([Buffer.from([1, 2, 3])]), 'a.wav')
  const created = await fetch(`${BASE}/cases`, { method: 'POST', body: fd }).then((r) => r.json())
  const caseId = created.case_id
  console.log('case yaratildi:', caseId)

  // 2) WS handshake
  const key = crypto.randomBytes(16).toString('base64')
  const sock = net.connect(PORT, '127.0.0.1')
  const seen = []
  await new Promise((resolve, reject) => {
    let buf = Buffer.alloc(0)
    let upgraded = false
    const timer = setTimeout(() => reject(new Error('timeout')), 8000)
    sock.on('connect', () => {
      sock.write(
        `GET /ws?case_id=${caseId} HTTP/1.1\r\n` +
          `Host: 127.0.0.1:${PORT}\r\n` +
          'Upgrade: websocket\r\n' +
          'Connection: Upgrade\r\n' +
          `Sec-WebSocket-Key: ${key}\r\n` +
          'Sec-WebSocket-Version: 13\r\n\r\n'
      )
    })
    sock.on('data', (chunk) => {
      buf = Buffer.concat([buf, chunk])
      if (!upgraded) {
        const i = buf.indexOf('\r\n\r\n')
        if (i === -1) return
        const head = buf.slice(0, i).toString()
        if (!head.includes('101')) return reject(new Error('handshake fail: ' + head))
        upgraded = true
        buf = buf.slice(i + 4)
      }
      buf = decodeFrames(buf, (txt) => {
        try {
          const ev = JSON.parse(txt)
          seen.push(ev.type)
          if (ev.type === 'explanation_token') process.stdout.write(String(ev.data.token))
          if (ev.type === 'case_done') {
            clearTimeout(timer)
            resolve()
          }
        } catch {
          /* skip */
        }
      })
    })
    sock.on('error', reject)
  })
  sock.destroy()

  console.log('\n--- ko\'rilgan event tartibi ---')
  console.log(seen.join(' -> '))
  const need = ['ready', 'tier1_done', 'stt_partial', 'stt_done', 'explanation_token', 'explanation_done', 'case_done']
  const missing = need.filter((t) => !seen.includes(t))
  if (missing.length) {
    console.error('FAIL — yetishmaydi:', missing.join(', '))
    process.exit(1)
  }
  console.log('PASS — barcha jonli eventlar real-time keldi ✅')
  process.exit(0)
}

main().catch((e) => {
  console.error('smoke xato:', e)
  process.exit(1)
})
