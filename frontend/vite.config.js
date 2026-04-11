import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  envDir: '../',
  build: {
    // Output directly into the FastAPI static directory so the single
    // Docker container can serve both the API and the React SPA.
    outDir: '../backend/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // Local dev: proxy all /api calls to FastAPI on :8000
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
