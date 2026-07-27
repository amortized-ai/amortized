import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  ArrowLeft,
  ExternalLink,
  Cpu,
  Database,
  Scale,
  CircleCheck,
  CircleX,
  Clock,
  Loader2,
  Trash2,
} from "lucide-react"
import { useModelRunData, useModelJobs } from "../api/use-models"
import { formatDate, cn } from "@/lib/utils"
import { EditableTitle } from "@/components/editable-title"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import { useNavigate } from "react-router"
import type { ModelRecord, Job } from "@/types/api"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"

const lossConfig = {
  loss: { label: "Loss", color: "var(--color-rh-red)" },
} satisfies ChartConfig

const lrConfig = {
  learning_rate: { label: "Learning Rate", color: "var(--color-rh-teal)" },
} satisfies ChartConfig

function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (minutes < 60) return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string; Icon: typeof CircleCheck }> = {
    succeeded: { label: "Succeeded", cls: "bg-rh-green-light text-rh-green-dark dark:bg-rh-green-dark/40 dark:text-rh-green", Icon: CircleCheck },
    failed: { label: "Failed", cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-400", Icon: CircleX },
    running: { label: "Running", cls: "bg-rh-blue-light text-rh-blue-dark dark:bg-rh-blue-dark/40 dark:text-rh-blue", Icon: Loader2 },
    pending: { label: "Pending", cls: "bg-muted text-muted-foreground", Icon: Clock },
  }
  const s = map[status] ?? map.pending!
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold", s.cls)}>
      <s.Icon className={cn("h-3 w-3", status === "running" && "animate-spin")} />
      {s.label}
    </span>
  )
}

function JobRow({ job, label, icon: Icon, color }: { job: Job; label: string; icon: typeof Database; color: string }) {
  const navigate = useNavigate()
  return (
    <button
      type="button"
      onClick={() => void navigate(`/jobs?job=${encodeURIComponent(job.id)}`)}
      className="flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors hover:bg-muted/50 w-full"
    >
      <Icon className={cn("h-4 w-4 shrink-0", color)} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-muted-foreground font-mono truncate">{job.id.slice(0, 8)}</p>
      </div>
      <StatusBadge status={job.status} />
    </button>
  )
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="text-lg font-bold tabular-nums text-foreground">{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
    </div>
  )
}

interface ModelDetailProps {
  name: string
  versions: ModelRecord[]
  onBack: () => void
  onDelete?: () => void
}

export function ModelDetail({ name, versions, onBack, onDelete }: ModelDetailProps) {
  const latest = versions[0]!
  const { getName, setName } = useEntityNamesStore()
  const { data: runData, isLoading: runLoading } = useModelRunData(latest.run_id || null)
  const { data: jobsData } = useModelJobs(latest.run_id || null)

  const baseModel = runData?.params["model_path"] || runData?.params["model_name_or_path"] || runData?.params["_name_or_path"] || "—"
  const method = runData?.params["algorithm"] || runData?.params["training_method"] ||
    (runData?.tags["job_type"] === "training" ? "LoRA SFT" : null) || "—"
  const epochs = runData?.finalMetrics["epoch"] || runData?.params["num_epochs"] || runData?.params["num_train_epochs"] || "—"
  const finalLoss = runData?.finalMetrics["loss"] ?? runData?.finalMetrics["train_loss"]
  const learningRate = runData?.finalMetrics["learning_rate"] || runData?.params["learning_rate"]
  const trainingDuration =
    runData?.run.info.end_time && runData?.run.info.start_time
      ? formatDuration(runData.run.info.end_time - runData.run.info.start_time)
      : "—"
  const runStatus = runData?.run.info.status ?? "UNKNOWN"

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <EditableTitle
            value={getName(name) || name}
            className="text-xl font-bold"
            onSave={(newName) => setName(name, newName)}
          />
          {latest.aliases.map((alias) => (
            <Badge key={alias} variant="outline" className="font-mono text-xs">
              @{alias}
            </Badge>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {latest.run_id && (
            <Button variant="outline" size="sm" asChild>
              <a
                href={runData?.run.info.experiment_id
                  ? `/mlflow/#/experiments/${runData.run.info.experiment_id}/runs/${latest.run_id}`
                  : `/mlflow/#/runs/${latest.run_id}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="h-4 w-4 mr-1" />
                View in MLflow
              </a>
            </Button>
          )}
          {onDelete && (
            <Button variant="outline" size="sm" onClick={onDelete}>
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <MetricCard
              label="Final Loss"
              value={finalLoss !== undefined ? finalLoss.toFixed(4) : "—"}
              sub={runStatus === "FINISHED" ? "Training complete" : runStatus.toLowerCase()}
            />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <MetricCard
              label="Epochs"
              value={String(epochs)}
              sub={learningRate != null ? `lr: ${typeof learningRate === "number" ? learningRate.toExponential(1) : learningRate}` : undefined}
            />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <MetricCard
              label="Training Time"
              value={trainingDuration}
              sub={`Version v${latest.version}`}
            />
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <MetricCard
              label="Method"
              value={method === "LoRA SFT" ? method : method.replace(/_/g, " ").toUpperCase()}
              sub={formatDate(new Date(latest.created_at).toISOString())}
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {/* Model info */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm">Training Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            {runLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading run data...
              </div>
            ) : runData ? (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                <dt className="text-muted-foreground">Base Model</dt>
                <dd className="font-mono text-xs truncate" title={baseModel}>{baseModel}</dd>
                <dt className="text-muted-foreground">Algorithm</dt>
                <dd>{method}</dd>
                <dt className="text-muted-foreground">Epochs</dt>
                <dd>{runData.params["num_train_epochs"] || runData.params["num_epochs"] || epochs}</dd>
                <dt className="text-muted-foreground">Learning Rate</dt>
                <dd className="font-mono text-xs">{runData.params["learning_rate"] || "—"}</dd>
                <dt className="text-muted-foreground">Batch Size</dt>
                <dd>{runData.params["per_device_train_batch_size"] || runData.params["effective_batch_size"] || runData.params["batch_size"] || runData.params["micro_batch_size"] || "—"}</dd>
                <dt className="text-muted-foreground">Max Seq Length</dt>
                <dd>{runData.params["max_seq_length"] || runData.params["max_seq_len"] || "—"}</dd>
                <dt className="text-muted-foreground">Run ID</dt>
                <dd className="font-mono text-xs truncate" title={latest.run_id}>{latest.run_id || "—"}</dd>
                <dt className="text-muted-foreground">Source</dt>
                <dd className="font-mono text-xs truncate" title={latest.source}>{latest.source || "—"}</dd>
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground">No run data available.</p>
            )}
          </CardContent>
        </Card>

        {/* Associated jobs */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Associated Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {jobsData?.sdgJob && (
                <JobRow job={jobsData.sdgJob} label="Data Generation" icon={Database} color="text-rh-purple" />
              )}
              {jobsData?.trainingJob && (
                <JobRow job={jobsData.trainingJob} label="Training" icon={Cpu} color="text-rh-blue" />
              )}
              {jobsData?.evalJobs.map((j) => (
                <JobRow key={j.id} job={j} label="Evaluation" icon={Scale} color="text-rh-teal" />
              ))}
              {!jobsData?.trainingJob && !jobsData?.sdgJob && jobsData?.evalJobs.length === 0 && (
                <p className="text-sm text-muted-foreground">No linked jobs found.</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Training charts */}
      {runData && (
        <div className="grid gap-4 md:grid-cols-2">
          {runData.histories["loss"] && runData.histories["loss"].length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Training Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer config={lossConfig} className="min-h-[200px] w-full">
                  <LineChart data={runData.histories["loss"]} accessibilityLayer>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="step" className="text-xs" />
                    <YAxis className="text-xs" />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name="Loss"
                      stroke="var(--color-rh-red)"
                      strokeWidth={2}
                      dot={runData.histories["loss"]!.length < 20}
                    />
                  </LineChart>
                </ChartContainer>
              </CardContent>
            </Card>
          )}

          {runData.histories["learning_rate"] && runData.histories["learning_rate"].length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Learning Rate Schedule</CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer config={lrConfig} className="min-h-[200px] w-full">
                  <LineChart data={runData.histories["learning_rate"]} accessibilityLayer>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="step" className="text-xs" />
                    <YAxis className="text-xs" />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name="Learning Rate"
                      stroke="var(--color-rh-teal)"
                      strokeWidth={2}
                      dot={runData.histories["learning_rate"]!.length < 20}
                    />
                  </LineChart>
                </ChartContainer>
              </CardContent>
            </Card>
          )}

          {(!runData.histories["loss"] || runData.histories["loss"].length === 0) &&
           (!runData.histories["learning_rate"] || runData.histories["learning_rate"].length === 0) && (
            <Card className="md:col-span-2">
              <CardContent className="p-8 flex items-center justify-center text-sm text-muted-foreground">
                No training metrics history available. Metrics are logged during training runs.
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Version history */}
      {versions.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Version History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {versions.map((v) => (
                <div key={v.version} className="flex items-center justify-between text-sm">
                  <span className="font-mono">v{v.version}</span>
                  <div className="flex items-center gap-2">
                    {v.aliases.map((alias) => (
                      <Badge key={alias} variant="outline" className="text-xs">
                        @{alias}
                      </Badge>
                    ))}
                    <span className="text-muted-foreground">
                      {formatDate(new Date(v.created_at).toISOString())}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
