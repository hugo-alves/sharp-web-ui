import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Set to true to use local backend, false to use deployed Vercel proxy
const USE_LOCAL_BACKEND = false;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api/proxy': USE_LOCAL_BACKEND
        ? {
            target: 'http://localhost:8000',
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/proxy/, '/api'),
          }
        : {
            target: 'https://frontend-hugoalves-projects.vercel.app',
            changeOrigin: true,
            secure: true,
          },
    },
  },
})
