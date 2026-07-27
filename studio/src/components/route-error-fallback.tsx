import { Button } from "@/components/ui/button"
import { AlertTriangle } from "lucide-react"
import type { FallbackProps } from "react-error-boundary"

export function RouteErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20">
      <AlertTriangle className="h-10 w-10 text-destructive" />
      <h2 className="text-lg font-semibold">Something went wrong</h2>
      <p className="max-w-md text-center text-sm text-muted-foreground">
        {error instanceof Error ? error.message : "An unexpected error occurred."}
      </p>
      <Button onClick={resetErrorBoundary}>Try Again</Button>
    </div>
  )
}
