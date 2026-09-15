import { useEffect, useState } from "react"
import { useNavigate } from "react-router"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { X, ClipboardCheck, Briefcase, ArrowRight, Database, Cpu } from "lucide-react"
import type { EvaluationEntry, EvaluationGroup } from "@/lib/api-client"
import { useDatasets, fetchDatasetByRun } from "@/features/datasets/api/use-datasets"
import { useModels } from "@/features/models/api/use-models"
import type { DatasetRecord } from "@/types/api"

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
  const { data: datasets } = useDatasets()
  const { data: models } = useModels()
  // Datasets with evals may be soft-deleted (absent from the list) — fetch
  // the run directly so the info card still shows its source details.
  const [fetchedDataset, setFetchedDataset] = useState<DatasetRecord | null>(null)
  const dataset =
    datasets?.find((d) => d.run_id === group?.dataset.run_id) ?? fetchedDataset
  useEffect(() => {
    setFetchedDataset(null)
    if (!group || dataset) return
    let cancelled = false
    fetchDatasetByRun(group.dataset.run_id)
      .then((rec) => {
        if (!cancelled) setFetchedDataset(rec)
      })
      .catch(() => {
        // run truly gone — card falls back to name only
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch only when the group's dataset changes and it is not in the list
  }, [group?.dataset.run_id, dataset === undefined])
  if (!group) return null

  // Models are listed under registered names but displayed as mdl-* —
  // resolve either way, and only link when the Models tab has the model.
  const resolveModel = (name: string) =>
    models?.find(
      (m) => m.name === name || m.tags?.model_display_name === name,
    )
  const metricRows = buildMetricRows(group)
  // Multiple eval runs of the same model (same dataset + metric set)
  // merge into one column: mean across runs, ± std when >1 value.
  const modelColumns = mergeEvalsByModel(group.evals)

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
                {modelColumns.length} model{modelColumns.length === 1 ? "" : "s"} ·{" "}
                {group.metric_names.length} metric{group.metric_names.length === 1 ? "" : "s"}
                {group.evals.length > modelColumns.length &&
                  ` · ${group.evals.length} runs`}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Comparison table */}
        <div className="flex-1 min-h-0 overflow-auto px-6 py-4">
          {/* Dataset source info */}
          <div className="rounded-xl border bg-card p-4 mb-4">
            <div className="flex items-start justify-between gap-4 py-1.5">
              <span className="text-sm text-muted-foreground shrink-0 inline-flex items-center gap-1.5">
                <Database className="h-3.5 w-3.5" />
                Dataset
              </span>
              <button
                type="button"
                data-testid="eval-dataset-link"
                onClick={() => {
                  onOpenChange(false)
                  setTimeout(() => navigate(`/datasets?run=${encodeURIComponent(group.dataset.run_id)}`), 200)
                }}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline text-right"
              >
                {group.dataset.name}
                <ArrowRight className="h-3 w-3 shrink-0" />
              </button>
            </div>
            {dataset && (
              <>
                {dataset.tags["dataset_topic"] !== undefined && (
                  <div className="flex items-start justify-between gap-4 py-1.5">
                    <span className="text-sm text-muted-foreground shrink-0">Topic</span>
                    <span className="text-sm text-right">{dataset.tags["dataset_topic"]}</span>
                  </div>
                )}
                {dataset.tags["num_samples"] !== undefined && (
                  <div className="flex items-start justify-between gap-4 py-1.5">
                    <span className="text-sm text-muted-foreground shrink-0">Samples</span>
                    <span className="text-sm text-right">{dataset.tags["num_samples"]}</span>
                  </div>
                )}
                {dataset.tags["teacher_model"] !== undefined && (
                  <div className="flex items-start justify-between gap-4 py-1.5">
                    <span className="text-sm text-muted-foreground shrink-0">Teacher Model</span>
                    <span className="text-sm text-right font-mono text-xs">
                      {dataset.tags["teacher_model"]}
                    </span>
                  </div>
                )}
                {dataset.tags["job_id"] !== undefined && (
                  <div className="flex items-start justify-between gap-4 py-1.5">
                    <span className="text-sm text-muted-foreground shrink-0">Source Job</span>
                    <button
                      type="button"
                      data-testid="eval-dataset-job-link"
                      onClick={() => {
                        onOpenChange(false)
                        setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(dataset.tags["job_id"]!)}`), 200)
                      }}
                      className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                    >
                      {dataset.tags["job_id"]!.slice(0, 8)}
                      <ArrowRight className="h-3 w-3" />
                    </button>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="rounded-xl border bg-card overflow-hidden">
            <table className="w-full min-w-max text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left font-medium px-4 py-3 sticky left-0 bg-muted/50 z-10 min-w-[180px]">
                    Metric
                  </th>
                  {modelColumns.map((col) => (
                    <th key={col.model} className="text-left font-medium px-4 py-3 min-w-[200px]">
                      <div className="flex flex-col gap-1">
                        {resolveModel(col.model) ? (
                          <button
                            type="button"
                            data-testid={`eval-model-link-${col.model}`}
                            onClick={() => {
                              onOpenChange(false)
                              setTimeout(() => navigate(`/models?name=${encodeURIComponent(col.model)}`), 200)
                            }}
                            className="inline-flex items-center gap-1 font-semibold truncate text-left hover:underline"
                            title={col.model}
                          >
                            <Cpu className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                            {col.model}
                          </button>
                        ) : (
                          <span
                            className="inline-flex items-center gap-1 font-semibold truncate text-left"
                            title={`${col.model} (not in the Models tab)`}
                          >
                            {col.model}
                          </span>
                        )}
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                          {col.runs.map((e) => (
                            <button
                              key={e.job_id}
                              type="button"
                              data-testid={`eval-job-link-${e.job_id}`}
                              onClick={() => {
                                onOpenChange(false)
                                setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(e.job_id)}`), 200)
                              }}
                              className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <Briefcase className="h-3 w-3" />
                              <span className="font-mono">{e.job_id.slice(0, 8)}</span>
                            </button>
                          ))}
                        </div>
                        <div className="flex items-center gap-1.5">
                          <StatusDot
                            status={
                              col.runs.every((e) => e.status === "succeeded")
                                ? "succeeded"
                                : (col.runs[0]?.status ?? "unknown")
                            }
                          />
                          <span className="text-xs text-muted-foreground">
                            {col.runs.every((e) => e.status === "succeeded")
                              ? col.runs.length > 1
                                ? `${col.runs.length} runs · mean ± std`
                                : col.runs[0]?.num_samples != null
                                  ? `${col.runs[0].num_samples} samples`
                                  : "done"
                              : (col.runs[0]?.status ?? "")}
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
                    <td className="px-4 py-3 font-medium sticky left-0 bg-card z-10">
                      {row.name}
                    </td>
                    {modelColumns.map((col) => {
                      const cell = row.cells[col.model]
                      return (
                        <td key={col.model} className="px-4 py-3">
                          {cell == null ? (
                            <span className="text-muted-foreground/50">--</span>
                          ) : (
                            <div className="flex items-center gap-2">
                              <span
                                className="font-medium tabular-nums"
                                title={
                                  cell.values.length > 1
                                    ? `runs: ${cell.values.map((v) => formatScore(v)).join(", ")}`
                                    : undefined
                                }
                              >
                                {formatScore(cell.mean)}
                                {cell.std != null && (
                                  <span className="ml-1 text-xs font-normal text-muted-foreground">
                                    ±{formatScore(cell.std!)}
                                  </span>
                                )}
                              </span>
                              <ScoreBar value={cell.mean} />
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

interface ModelColumn {
  model: string
  runs: EvaluationEntry[]
}

/** Group eval runs by model id, first-appearance order. */
function mergeEvalsByModel(evals: EvaluationEntry[]): ModelColumn[] {
  const byModel = new Map<string, EvaluationEntry[]>()
  for (const e of evals) {
    const runs = byModel.get(e.model)
    if (runs) runs.push(e)
    else byModel.set(e.model, [e])
  }
  return [...byModel.entries()].map(([model, runs]) => ({ model, runs }))
}

interface ScoreStat {
  mean: number
  std: number | null
  values: number[]
}

function scoreStat(values: number[]): ScoreStat {
  const n = values.length
  const mean = values.reduce((a, b) => a + b, 0) / n
  let std: number | null = null
  if (n > 1) {
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
    std = Math.sqrt(variance)
  }
  return { mean, std, values }
}

function buildMetricRows(group: EvaluationGroup): {
  name: string
  cells: Record<string, ScoreStat | null>
}[] {
  const names = new Set<string>()
  for (const e of group.evals) {
    for (const key of Object.keys(e.scores)) names.add(key)
  }
  // Keep the group's declared metric order first, then any extras
  const ordered = [
    ...group.metric_names.filter((n) => names.has(n)),
    ...[...names].filter((n) => !group.metric_names.includes(n)).sort(),
  ]
  const columns = mergeEvalsByModel(group.evals)
  return ordered.map((name) => ({
    name,
    cells: Object.fromEntries(
      columns.map((col) => {
        const values = col.runs
          .filter((e) => e.status === "succeeded")
          .map((e) => e.scores[name])
          .filter((v): v is number => v != null)
        return [col.model, values.length ? scoreStat(values) : null]
      }),
    ),
  }))
}
