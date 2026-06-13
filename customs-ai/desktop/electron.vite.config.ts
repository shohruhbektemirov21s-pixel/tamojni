import { resolve } from 'path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'

// electron-vite: main / preload / renderer uchun alohida build.
// Renderer faqat HTTP/REST orqali backend bilan gaplashadi (ADR-005).
export default defineConfig({
  main: {
    build: {
      outDir: 'out/main',
      lib: { entry: resolve('src/main/index.ts') }
    }
  },
  preload: {
    build: {
      outDir: 'out/preload',
      lib: { entry: resolve('src/preload/index.ts') }
    }
  },
  renderer: {
    root: 'src/renderer',
    plugins: [react()],
    server: { port: 5273, strictPort: true },
    build: {
      outDir: 'out/renderer',
      rollupOptions: { input: resolve('src/renderer/index.html') }
    }
  }
})
