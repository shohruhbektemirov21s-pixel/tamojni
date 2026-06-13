# Bojxona Operator — Desktop (Electron + React + TS)

Offline Bojxona AI tizimining **operator desktop ilovasi**. Faqat backend REST/SSE
API'sini iste'mol qiladi (ADR-005) — modellarga to'g'ridan-to'g'ri tegmaydi.

## Tamoyillar (UI'da aks etadi)

1. **AI qaror qabul qilmaydi** — signal + tushuntirish beradi, qaror operatorda.
2. **Shaffoflik** — har qism provenance bilan: `⚙ Deterministik` (risk) vs `🧠 AI` (explanation),
   `available` flaglari, `degraded` ochiq ko'rsatiladi.
3. **"Topilmadi ≠ xavfsiz"** — LOW risk yoki bo'sh detektsiyada ham eslatma chiqadi.
4. **Endpoint sozlanadigan** — Sozlamalar'da 127.0.0.1 → LAN server IP (Faza 2).
5. **Offline by design** — CSP tashqi resurslarni bloklaydi.

## Ishga tushirish

```bash
npm install

# Variant A — mock backend bilan (real backend tayyor bo'lmasa):
npm run mock        # 1-terminal: §7 taqlidi, http://127.0.0.1:8787
npm run dev         # 2-terminal: Electron app
#   -> App: ⚙ Sozlamalar -> endpoint = http://127.0.0.1:8787

# yoki bitta buyruq bilan ikkalasi:
npm run dev:full

# Variant B — real backend (127.0.0.1:8000) — default endpoint, sozlamasiz ishlaydi.
```

## Build (Windows)

```bash
npm run build         # typecheck + electron-vite build
npm run build:win     # + electron-builder (Windows installer)
```

## Struktura

```
src/
  main/         Electron main process (oyna, offline)
  preload/      contextBridge (minimal)
  shared/       §7/§7.1 API tiplari (yagona haqiqat manbai)
  renderer/src/
    api/        config (endpoint) + HTTP/SSE klient + ApiError
    components/ ImageWithDetections (bbox overlay), RiskPanel (factors),
                ProvenanceBadge, ProgressTracker (SSE), DecisionBar, ...
    screens/    SubmitScreen, CaseView (progress + §7.1 render + qaror)
mock-server/    Express §7 taqlidi (SSE, multipart, xato envelope, §7.1)
```

## Mock ssenariylari

Har yangi case ketma-ket aylanadi: **HIGH → MEDIUM → LOW(bo'sh) → DEGRADED** —
shu bilan barcha UI holatlari (alert, factors, "topilmadi", provenance/degraded)
ko'rsatiladi.
