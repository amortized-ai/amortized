import { useNavigate } from "react-router"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { X, ClipboardCheck, Briefcase } from "lucide-react"
import type { EvaluationGroup } from "@/lib/api-client"

interface EvaluationDetailPanelProps {
  group: EvaluationGroup | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function EvaluationDetailPanel({
  group,
  open,
  onOpenChange,
}: EvaluationDetailPanelProps) {
  const navigate = useNavigate()
  if (!group) return null

  const metricRows = buildMetricRows(group)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        variant="panel"
        showCloseButton={false}
        className="!max-w-[1000px] !w-[1000px] !h-[80vh] flex flex-col !gap-0 p-0 overflow-hidden rounded-xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e6f7f2] text-[#0d7a5f] dark:bg-[#003326]/40 dark:text-[#3fbf9a]">
              <ClipboardCheck className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <DialogTitle className="text-base font-semibold truncate">
                {group.dataset.name}
              </DialogTitle>
              <p className="text-xs text-muted-foreground">
                {group.evals.length} model{group.evals.length === 1 ? "" : "s"} ·{" "}
                {group.metric_names.length} metric{group.metric_names.length === 1 ? "" : "s"}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Comparison table */}
        <div className="flex-1 overflow-auto px-6 py-4">
          <div className="rounded-xl border bg-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left font-medium px-4 py-3 sticky left-0 bg-muted/50 min-w-[180px]">
                    Metric
                  </th>
                  {group.evals.map((e) => (
                    <th key={e.job_id} className="text-left font-medium px-4 py-3 min-w-[200px]">
                      <div className="flex flex-col gap-1">
                        <span className="font-semibold truncate" title={e.model}>
                          {e.model}
                        </span>
                        <button
                          type="button"
                          data-testid={`eval-job-link-${e.job_id}`}
                          onClick={() => {
                            onOpenChange(false)
                            setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(e.job_id)}`), 200)
                          }}
                          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-fit"
                        >
                          <Briefcase className="h-3 w-3" />
                          <span className="font-mono">{e.job_id.slice(0, 8)}</span>
                        </button>
                        <div className="flex items-center gap-1.5">
                          <StatusDot status={e.status} />
                          <span className="text-xs text-muted-foreground">
                            {e.status === "succeeded"
                              ? e.num_samples != null
                                ? `${e.num_samples} samples`
                                : "done"
                              : e.status}
                          </span>
                        </div>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metricRows.map((row) => (
                  <tr key={row.name} className="border-b last:border-b-0">
                    <td className="px-4 py-3 font-medium sticky left-0 bg-card">
                      {row.name}
                    </td>
                    {group.evals.map((e) => {
                      const cell = row.cells[e.job_id]
                      return (
                        <td key={e.job_id} className="px-4 py-3">
                          {cell == null ? (
                            <span className="text-muted-foreground/50">--</span>
                          ) : (
                            <div className="flex items-center gap-2">
                              <span className="font-medium tabular-nums">
                                {formatScore(cell)}
                              </span>
                              <ScoreBar value={cell} />
                            </div>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-3 text-xs text-muted-foreground">
            Scores are 0–1 (0–100%); higher is better. Job IDs link to the Jobs page.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === "succeeded"
      ? "bg-rh-green"
      : status === "running"
        ? "bg-rh-yellow"
        : status === "failed"
          ? "bg-rh-red"
          : "bg-muted-foreground/40"
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} aria-label={status} />
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="h-1.5 w-16 rounded-full bg-muted overflow-hidden">
      <div
        className="h-full rounded-full bg-[#0d7a5f] dark:bg-[#3fbf9a]"
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

function formatScore(value: number): string {
  return `${Math.round(value * 1000) / 10}%`
}

function buildMetricRows(group: EvaluationGroup): { name: string; cells: Record<string, number | null> }[] {
  const names = new Set<string>()
  for (const e of group.evals) {
    for (const key of Object.keys(e.scores)) names.add(key)
  }
  // Keep the group's declared metric order first, then any extras
  const ordered = [
    ...group.metric_names.filter((n) => names.has(n)),
    ...[...names].filter((n) => !group.metric_names.includes(n)).sort(),
  ]
  return ordered.map((name) => ({
    name,
    cells: Object.fromEntries(
      group.evals.map((e) => [e.job_id, e.scores[name] ?? null]),
    ),
  }))
}
