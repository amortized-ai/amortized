import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import "@rhds/elements/rh-alert/rh-alert.js"
import "@rhds/elements/rh-skeleton/rh-skeleton.js"
import App from "./App"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
