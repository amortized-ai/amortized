import { useNavigate } from "react-router"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { X, ClipboardCheck, Briefcase } from "lucide-react"
import type { EvaluationGroup, EvaluationEntry } from "@/lib/api-client"

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

  const metricNames = buildMetricNames(group)

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
                {metricNames.length} metric{metricNames.length === 1 ? "" : "s"}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Comparison table — one row per model, one column per metric.
            Scales to many models (vertical scroll) while metrics stay fixed. */}
        <div className="flex-1 overflow-auto px-6 py-4">
          <div className="rounded-xl border bg-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left font-medium px-4 py-3 min-w-[240px]">
                    Model
                  </th>
                  {metricNames.map((name) => (
                    <th key={name} className="text-left font-medium px-4 py-3 min-w-[140px]">
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {group.evals.map((e) => (
                  <tr key={e.job_id} className="border-b last:border-b-0 hover:bg-muted/30">
                    <td className="px-4 py-3">
                      <ModelCell entry={e} onOpenJob={() => {
                        onOpenChange(false)
                        setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(e.job_id)}`), 200)
                      }} />
                    </td>
                    {metricNames.map((name) => (
                      <ScoreCell
                        key={name}
                        value={e.scores[name] ?? null}
                        best={isBest(group, name, e.scores[name] ?? null)}
                      />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-3 text-xs text-muted-foreground">
            Scores are 0–100%; higher is better. Best score per metric is highlighted. Click a
            job ID to open the job.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function ModelCell({ entry, onOpenJob }: { entry: EvaluationEntry; onOpenJob: () => void }) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <StatusDot status={entry.status} />
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-semibold truncate" title={entry.model}>
            {entry.model}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <button
            type="button"
            data-testid={`eval-job-link-${entry.job_id}`}
            onClick={(ev) => {
              ev.stopPropagation()
              onOpenJob()
            }}
            className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
          >
            <Briefcase className="h-3 w-3" />
            <span className="font-mono">{entry.job_id.slice(0, 8)}</span>
          </button>
          {entry.status === "succeeded" && entry.num_samples != null && (
            <span>· {entry.num_samples} samples</span>
          )}
          {entry.status !== "succeeded" && <span>· {entry.status}</span>}
        </div>
      </div>
    </div>
  )
}

function ScoreCell({ value, best }: { value: number | null; best: boolean }) {
  if (value == null) {
    return (
      <td className="px-4 py-3">
        <span className="text-muted-foreground/50">--</span>
      </td>
    )
  }
  return (
    <td className="px-4 py-3">
      <div className="flex items-center gap-2">
        <span
          className={`font-medium tabular-nums ${best ? "text-[#0d7a5f] dark:text-[#3fbf9a] font-semibold" : ""}`}
        >
          {formatScore(value)}
        </span>
        <ScoreBar value={value} highlight={best} />
      </div>
    </td>
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
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${color} shrink-0`} aria-label={status} />
}

function ScoreBar({ value, highlight }: { value: number; highlight: boolean }) {
  const pct = Math.round(value * 100)
  return (
    <div className="h-1.5 w-14 rounded-full bg-muted overflow-hidden">
      <div
        className={`h-full rounded-full ${highlight ? "bg-[#0d7a5f] dark:bg-[#3fbf9a]" : "bg-muted-foreground/40"}`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

function formatScore(value: number): string {
  return `${Math.round(value * 1000) / 10}%`
}

/** Union of the group's declared metrics and every score key present. */
function buildMetricNames(group: EvaluationGroup): string[] {
  const names = new Set<string>()
  for (const e of group.evals) {
    for (const key of Object.keys(e.scores)) names.add(key)
  }
  return [
    ...group.metric_names.filter((n) => names.has(n)),
    ...[...names].filter((n) => !group.metric_names.includes(n)).sort(),
  ]
}

/** True when `value` is the highest non-null score for `metric` in the group. */
function isBest(group: EvaluationGroup, metric: string, value: number | null): boolean {
  if (value == null) return false
  let best: number | null = null
  for (const e of group.evals) {
    const v = e.scores[metric] ?? null
    if (v != null && (best == null || v > best)) best = v
  }
  return best != null && value >= best
}
