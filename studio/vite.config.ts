import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const proxy = process.env.VITE_PROXY_TARGET

// Public base path. Defaults to "/" (standalone/kind). Set VITE_BASE_PATH
// (e.g. "/amortized-studio-embed") to serve the SPA under a subpath when
// embedded behind the RHOAI dashboard proxy. Drives asset URLs, the router
// basename, and the API base (all via import.meta.env.BASE_URL).
function normalizeBase(b?: string): string {
  if (!b || b === "/") return "/"
  const withLead = b.startsWith("/") ? b : `/${b}`
  return withLead.endsWith("/") ? withLead : `${withLead}/`
}
const base = normalizeBase(process.env.VITE_BASE_PATH)

export default defineConfig({
  base,
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
