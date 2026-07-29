import { AlertTriangle, CheckCircle, Loader2, X } from "lucide-react"
import type { SessionStatus } from "@/stores/chat-store"

interface SessionStatusBannerProps {
  status: SessionStatus
  onDismiss: () => void
}

export function SessionStatusBanner({ status, onDismiss }: SessionStatusBannerProps) {
  if (status === "unknown" || status === "connected") return null

  if (status === "reconnecting") {
    return (
      <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground bg-muted/50 rounded-md mx-4 mt-2">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>Reconnecting to session...</span>
      </div>
    )
  }

  if (status === "restored") {
    return (
      <div className="flex items-center justify-between gap-2 px-3 py-2 text-xs bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 rounded-md mx-4 mt-2">
        <div className="flex items-center gap-2">
          <CheckCircle className="h-3 w-3 flex-shrink-0" />
          <span>Session restored with full conversation history.</span>
        </div>
        <button onClick={onDismiss} className="flex-shrink-0 hover:opacity-70">
          <X className="h-3 w-3" />
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 text-xs bg-amber-500/10 text-amber-700 dark:text-amber-400 rounded-md mx-4 mt-2">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-3 w-3 flex-shrink-0" />
        <span>Context was rebuilt from chat history. The agent may not recall all details.</span>
      </div>
      <button onClick={onDismiss} className="flex-shrink-0 hover:opacity-70">
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}
