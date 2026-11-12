import { AlertTriangle } from "lucide-react"
import type { ModelCheckResult } from "@/types/api"

// Why an endpoint couldn't be verified, in words. Unverified checks don't block the job,
// so the user needs to see them rather than assume the endpoint is known to work.
const REASONS: Record<string, string> = {
  rate_limited: "the endpoint is rate-limiting requests",
  checker_unavailable: "the model check service was unavailable",
  checker_busy: "the model check service was busy",
  budget_exhausted: "the check ran out of time",
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
      {unverified.map((check) => (
        <p
          key={`${check.label}-${check.model}`}
          className="flex items-start gap-1.5 text-[11px] text-amber-700 dark:text-amber-400"
        >
          <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
          <span>{describe(check)}</span>
        </p>
      ))}
    </div>
  )
}
