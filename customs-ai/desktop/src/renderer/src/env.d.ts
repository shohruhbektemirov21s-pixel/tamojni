/// <reference types="vite/client" />

// preload orqali ochilgan minimal API (main/preload/index.ts).
interface AppInfo {
  platform: string
  versions: { electron: string; chrome: string; node: string }
}

interface Window {
  appInfo?: AppInfo
}
