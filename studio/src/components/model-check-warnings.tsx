import { AlertTriangle } from "lucide-react"
import type { ModelCheckResult } from "@/types/api"

// Why an endpoint couldn't be verified, in words. Unverified checks don't block the job,
// so the user needs to see them rather than assume the endpoint is known to work.
const REASONS: Record<string, string> = {
  rate_limited: "the endpoint is rate-limiting requests",
  deadline: "it didn't answer in time (a cold endpoint may still serve the job)",
  server_error: "the endpoint returned server errors",
  unreachable: "it couldn't be reached from the server (the job runs elsewhere and may still reach it)",
  checker_unavailable: "the model check service was unavailable",
  checker_busy: "too many model checks were running",
  budget_exhausted: "the check ran out of time",
  internal_error: "the model check hit an internal error",
}

function describe(check: ModelCheckResult): string {
  const label = check.label.charAt(0).toUpperCase() + check.label.slice(1)
  const why = REASONS[check.reason] ?? check.reason.replaceAll("_", " ")
  return `${label} ${check.model} not verified: ${why}.`
}

export function ModelCheckWarnings({ checks }: { checks?: ModelCheckResult[] }) {
  const unverified = (checks ?? []).filter((c) => c.outcome === "unverified")
  if (unverified.length === 0) return null
  return (
    <div className="mt-2 space-y-1" data-testid="model-check-warnings">
      {/* Keyed by position: two endpoints can share a label and model (e.g. repeated SDG
          aliases, or the same model under different credentials). */}
      {unverified.map((check, i) => (
        <p
          key={i}
          className="flex items-start gap-1.5 text-[11px] text-amber-700 dark:text-amber-400"
        >
          <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
          <span>{describe(check)}</span>
        </p>
      ))}
    </div>
  )
}
