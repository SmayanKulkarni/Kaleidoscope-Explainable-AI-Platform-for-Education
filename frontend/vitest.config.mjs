import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_API_URL': JSON.stringify('http://localhost:8000'),
    'import.meta.env.MODE':         JSON.stringify('test'),
    'import.meta.env.DEV':          JSON.stringify(false),
    'import.meta.env.PROD':         JSON.stringify(false),
    'import.meta.env.SSR':          JSON.stringify(false),
  },
  test: {
    environment:  'happy-dom',
    globals:      true,
    setupFiles:   ['./src/tests/setup.js'],
    clearMocks:   true,
  },
});
