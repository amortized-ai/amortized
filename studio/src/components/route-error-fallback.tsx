import { useState } from "react"
import { Button } from "@/components/ui/button"
import { AlertTriangle, ChevronDown } from "lucide-react"
import type { FallbackProps } from "react-error-boundary"

export function RouteErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  const [showDetails, setShowDetails] = useState(false)

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20">
      <AlertTriangle className="h-10 w-10 text-destructive" />
      <h2 className="text-lg font-semibold">This page ran into a problem</h2>
      <p className="max-w-md text-center text-sm text-muted-foreground">
        Try refreshing, or go back and try again.
      </p>
      <div className="flex items-center gap-2">
        <Button variant="outline" onClick={() => window.history.back()}>
          Go back
        </Button>
        <Button onClick={resetErrorBoundary}>Try again</Button>
      </div>
      {error instanceof Error && error.message && (
        <div className="mt-2">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronDown
              className={`h-3 w-3 transition-transform duration-200 ${showDetails ? "rotate-180" : ""}`}
            />
            Technical details
          </button>
          {showDetails && (
            <p className="mt-2 max-w-lg rounded-md bg-muted px-3 py-2 font-mono text-xs text-muted-foreground">
              {error.message}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
