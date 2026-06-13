// Endpoint SOZLANADIGAN (ADR-005, Faza 2: 127.0.0.1 -> server IP).
// localStorage'da saqlanadi; offline by design — boshqa hech qayerga chiqmaydi.

const STORAGE_KEY = 'customs.api.baseUrl'
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

export function getBaseUrl(): string {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v && v.trim() ? v.trim().replace(/\/+$/, '') : DEFAULT_BASE_URL
  } catch {
    return DEFAULT_BASE_URL
  }
}

export function setBaseUrl(url: string): void {
  const clean = url.trim().replace(/\/+$/, '')
  localStorage.setItem(STORAGE_KEY, clean || DEFAULT_BASE_URL)
}

export function resetBaseUrl(): void {
  localStorage.removeItem(STORAGE_KEY)
}

export { DEFAULT_BASE_URL }
