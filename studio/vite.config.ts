import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const proxy = process.env.VITE_PROXY_TARGET

export default defineConfig({
  // Asset base path. Root for standalone; set to /amortized-studio-embed at build time
  // (VITE_BASE_PATH) for the RHOAI dashboard embed so assets resolve under the
  // dashboard proxy prefix instead of the dashboard origin root.
  base: process.env.VITE_BASE_PATH || "/",
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
        target: proxy ?? "http://localhost:4096",
        changeOrigin: true,
        timeout: 120000,
        ...(proxy ? {} : { rewrite: (p: string) => p.replace(/^\/agent/, "") }),
      },
    },
  },
})
