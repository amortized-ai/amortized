import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const proxy = process.env.VITE_PROXY_TARGET

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: proxy ?? "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
      "/mlflow": {
        target: proxy ?? "http://localhost:5000",
        changeOrigin: true,
        ...(proxy ? {} : { rewrite: (p: string) => p.replace(/^\/mlflow/, "") }),
      },
      "/agent": {
        target: proxy ?? "http://localhost:8000",
        changeOrigin: true,
        timeout: 120000,
      },
    },
  },
})
