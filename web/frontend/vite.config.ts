import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api/proxy': {
        target: 'https://frontend-hugoalves-projects.vercel.app',
        changeOrigin: true,
        secure: true,
      },
    },
  },
})
