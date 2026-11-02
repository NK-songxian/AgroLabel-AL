import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 后端默认跑在 8000；把 /api 代理过去，前端一律用相对路径，
// 既省掉跨域配置，也让 vite build 出来的产物可以直接挂到任意反代后面。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
