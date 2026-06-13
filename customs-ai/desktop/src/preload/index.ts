import { contextBridge } from 'electron'

// Minimal va xavfsiz: hozircha renderer'ga faqat platforma ma'lumotini beramiz.
// Barcha backend muloqoti renderer'dan to'g'ridan-to'g'ri HTTP (ADR-005) orqali.
contextBridge.exposeInMainWorld('appInfo', {
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node
  }
})
