import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { EditableTitle } from "@/components/editable-title"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { ChevronDown, XCircle, AlertCircle, ArrowRight, X, Database, ExternalLink, Trash2, FileText } from "lucide-react"
import { Link, useNavigate } from "react-router"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { LogViewer } from "@/components/log-viewer"
import { JsonTreeViewer } from "@/components/json-tree-viewer"
import { StatusBadge } from "./status-badge"
import { JobTypeBadge } from "./job-type-badge"
import { TrainingMetricsChart } from "./training-metrics-chart"
import { formatDuration } from "../lib/format"
import { formatDate } from "@/lib/utils"
import { useCancelJob, useDeleteJob, useJobLogs, useJobMlflowMetrics } from "../api/use-jobs"
import { DeleteEntityDialog } from "@/components/delete-entity-dialog"
import type { Job } from "@/types/api"

function formatJobError(raw: string): { summary: string; isTruncated: boolean } {
  const lower = raw.toLowerCase()
  if (lower.includes("experiment") && lower.includes("does not match"))
    return { summary: "MLflow experiment mismatch — the document was converted successfully, but results couldn't be saved. This is a backend configuration issue.", isTruncated: false }
  if (/exit.*code 1|code 1.*error/i.test(raw) && raw.length < 80)
    return { summary: "The job process crashed. Check the Logs tab for the full error traceback.", isTruncated: false }
  if (raw.length > 200) {
    return { summary: raw.slice(0, 200) + "…", isTruncated: true }
  }
  return { summary: raw, isTruncated: false }
}

interface JobDetailPanelProps {
  job: Job | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function JobDetailPanel({ job, open, onOpenChange }: JobDetailPanelProps) {
  const cancelMutation = useCancelJob()
  const deleteMutation = useDeleteJob()
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const { getName, setName } = useEntityNamesStore()

  if (!open) return null

  const canCancel = job ? ["queued", "provisioning", "running"].includes(job.status) : false
  const canDelete = job ? ["succeeded", "failed", "cancelled"].includes(job.status) : false

  function handleCancel() {
    cancelMutation.mutate(job!.id)
    setCancelDialogOpen(false)
  }

  function handleDelete() {
    deleteMutation.mutate(job!.id, {
      onSuccess: () => {
        setDeleteDialogOpen(false)
        onOpenChange(false)
      },
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent variant="panel" showCloseButton={false} className="!max-w-[900px] !w-[900px] !h-[80vh] flex flex-col !gap-0 p-0 overflow-hidden rounded-xl">
        {!job ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
              <span className="text-sm">Loading job details...</span>
            </div>
          </div>
        ) : (<>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
          <div className="flex items-center gap-3 min-w-0 overflow-hidden">
            <DialogTitle asChild>
              <EditableTitle
                value={getName(job.id) || (job.metadata?.name as string) || job.id}
                className="text-base font-semibold truncate"
                onSave={(newName) => setName(job.id, newName)}
              />
            </DialogTitle>
            <div className="flex items-center gap-3 ml-2 shrink-0">
              <JobTypeBadge type={job.type} />
              <StatusBadge status={job.status} />
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 ml-4">
            {job.mlflow_run_id && (
              <Button variant="outline" size="sm" asChild>
                <a
                  href={`/mlflow/#/runs/${job.mlflow_run_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink className="mr-1 h-3.5 w-3.5" />
                  View in MLflow
                </a>
              </Button>
            )}
            {canCancel && (
              <Dialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="destructive" size="sm">
                    <XCircle className="mr-1 h-4 w-4" />
                    Cancel Job
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Cancel Job</DialogTitle>
                    <DialogDescription>
                      Are you sure you want to cancel this job? This action cannot be undone.
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setCancelDialogOpen(false)}>
                      Keep Running
                    </Button>
                    <Button variant="destructive" onClick={handleCancel}>
                      Cancel Job
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            )}
            {canDelete && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setDeleteDialogOpen(true)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
                <DeleteEntityDialog
                  open={deleteDialogOpen}
                  entityType="job"
                  entityName={getName(job!.id) || (job!.metadata?.name as string) || job!.id.slice(0, 8)}
                  onConfirm={handleDelete}
                  onCancel={() => setDeleteDialogOpen(false)}
                  isPending={deleteMutation.isPending}
                />
              </>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 opacity-70 hover:opacity-100"
              onClick={() => onOpenChange(false)}
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Close</span>
            </Button>
          </div>
        </div>

        {/* Tabs — fill remaining space */}
        <Tabs defaultValue="overview" className="flex flex-col flex-1 min-h-0">
          <div className="px-6 pt-3 shrink-0">
            <TabsList className="w-full">
              <TabsTrigger value="overview" className="flex-1 transition-all duration-200">Overview</TabsTrigger>
              <TabsTrigger value="logs" className="flex-1 transition-all duration-200">Logs</TabsTrigger>
              {job.type === "training" && <TabsTrigger value="metrics" className="flex-1 transition-all duration-200">Metrics</TabsTrigger>}
              <TabsTrigger value="config" className="flex-1 transition-all duration-200">Config</TabsTrigger>
            </TabsList>
          </div>

          {/* Each tab content is a direct flex child so the "logs" tab can
              fill the remaining height without a double scrollbar.
              Non-logs tabs use overflow-y-auto for their own scrolling. */}
          <TabsContent value="overview" className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4">
            <OverviewTab job={job} onClose={() => onOpenChange(false)} />
          </TabsContent>

          <TabsContent value="logs" className="mt-0 flex-1 min-h-0 flex flex-col px-6 py-4">
            <LogsTab job={job} />
          </TabsContent>

          {job.type === "training" && (
            <TabsContent value="metrics" className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4">
              <MetricsTab job={job} />
            </TabsContent>
          )}

          <TabsContent value="config" className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4">
            <ConfigTab job={job} />
          </TabsContent>
        </Tabs>
        </>)}
      </DialogContent>
    </Dialog>
  )
}

function OverviewTab({ job, onClose }: { job: Job; onClose: () => void }) {
  const navigate = useNavigate()

  return (
    <div className="space-y-0">
      {job.type === "sdg" && job.status === "succeeded" && job.mlflow_run_id && (
        <button
          type="button"
          onClick={() => {
            onClose()
            navigate(`/datasets?run=${encodeURIComponent(job.mlflow_run_id!)}`)
          }}
          className="group flex w-full items-center gap-3 rounded-xl border bg-card p-3.5 mb-4 text-left transition-all duration-200 hover:border-[#d0c5f4] hover:shadow-sm cursor-pointer dark:hover:border-[#21134d]"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#ece6ff] text-[#5e40be] transition-colors duration-200 group-hover:bg-[#d0c5f4] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
            <Database className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">View Generated Dataset</p>
            <p className="text-xs text-muted-foreground">Browse the synthetic training samples</p>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5" />
        </button>
      )}
      {job.type === "upload" && job.status === "succeeded" && job.mlflow_run_id && (
        <button
          type="button"
          onClick={() => {
            onClose()
            navigate(`/documents?doc=${encodeURIComponent(job.mlflow_run_id!)}`)
          }}
          className="group flex w-full items-center gap-3 rounded-xl border bg-card p-3.5 mb-4 text-left transition-all duration-200 hover:border-[#9ad8d8] hover:shadow-sm cursor-pointer dark:hover:border-[#004d4d]"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#daf2f2] text-[#147878] transition-colors duration-200 group-hover:bg-[#9ad8d8] dark:bg-[#003333]/40 dark:text-[#37a3a3]">
            <FileText className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">View Document</p>
            <p className="text-xs text-muted-foreground">Browse the parsed document and chunks</p>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5" />
        </button>
      )}
      <MetadataRow label="ID" value={job.id} mono />
      <MetadataRow label="Type" value={job.type} />
      <MetadataRow label="Status" value={job.status} />
      <MetadataRow label="Created" value={formatDate(job.created_at, { includeTime: true })} />
      {job.started_at && (
        <MetadataRow label="Started" value={formatDate(job.started_at, { includeTime: true })} />
      )}
      {job.completed_at && (
        <MetadataRow label="Completed" value={formatDate(job.completed_at, { includeTime: true })} />
      )}
      <MetadataRow
        label="Duration"
        value={formatDuration(job.started_at, job.completed_at)}
      />
      {job.recipe && <MetadataRow label="Recipe" value={job.recipe} />}
      {job.mlflow_run_id && <MetadataRow label="MLflow Run" value={job.mlflow_run_id} mono />}
      {job.parent_job_id && <MetadataRow label="Parent Job" value={job.parent_job_id} mono />}
      {job.k8s_namespace && <MetadataRow label="Namespace" value={job.k8s_namespace} />}
      {job.error && <ErrorSection job={job} />}
    </div>
  )
}

function ErrorSection({ job }: { job: Job }) {
  const [showFull, setShowFull] = useState(false)
  const { summary, isTruncated } = formatJobError(job.error ?? "")

  return (
    <div className="space-y-3 pt-3">
      <Alert variant="destructive">
        <AlertDescription>
          {showFull ? job.error : summary}
          {isTruncated && (
            <button
              className="ml-2 text-xs underline underline-offset-2 opacity-70 hover:opacity-100 transition-opacity"
              onClick={() => setShowFull(!showFull)}
            >
              {showFull ? "Show less" : "Show full error"}
            </button>
          )}
        </AlertDescription>
      </Alert>
      <FailureDiagnosis job={job} />
    </div>
  )
}

function FailureDiagnosis({ job }: { job: Job }) {
  const hints: { label: string; detail: string; link?: { to: string; text: string } }[] = []
  const error = (job.error ?? '').toLowerCase()
  const duration = job.started_at && job.completed_at
    ? new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()
    : null

  const isFast = duration !== null && duration < 10000

  if (job.type === 'sdg') {
    if (!job.mlflow_run_id) {
      hints.push({
        label: 'No MLflow connection',
        detail: 'SDG jobs need MLflow to track experiments. Start MLflow or set the tracking URI in backend config.',
        link: { to: '/settings', text: 'Check Settings' }
      })
    }
    hints.push({
      label: 'AI Gateway or local model required',
      detail: 'SDG needs an LLM to generate data. Point it at a local model (via api_base) or add a cloud provider gateway route.',
      link: { to: '/settings', text: 'Configure AI Gateway' }
    })
    if (isFast) {
      hints.push({
        label: 'Job failed instantly',
        detail: 'A 2-second failure usually means a missing dependency or configuration error, not a data issue.'
      })
    }
  }

  if (job.type === 'training') {
    if (isFast) {
      hints.push({
        label: 'No GPU detected',
        detail: 'Training requires a GPU. Check that CUDA is available or configure an SSH backend with GPU access.',
        link: { to: '/settings', text: 'Check Settings' }
      })
    }
    if (!job.mlflow_run_id) {
      hints.push({
        label: 'No MLflow connection',
        detail: 'Training jobs log metrics to MLflow. Ensure MLflow is running and the tracking URI is configured.'
      })
    }
  }

  if (job.type === 'upload') {
    if (error.includes('experiment') && error.includes('match')) {
      hints.push({
        label: 'MLflow experiment mismatch',
        detail: 'The document processor tried to log to an MLflow experiment that doesn\'t match the run. This is usually a backend configuration issue — check that the MLFLOW_EXPERIMENT_NAME environment variable matches the experiment used during upload.',
      })
    }
    if (!job.mlflow_run_id) {
      hints.push({
        label: 'No MLflow connection',
        detail: 'Document processing needs MLflow to store parsed content and chunks. Ensure MLflow is running and the tracking URI is configured.',
        link: { to: '/settings', text: 'Check Settings' }
      })
    }
    if (isFast && hints.length === 0) {
      hints.push({
        label: 'Job failed quickly',
        detail: 'The document processing job exited almost immediately. This usually means a configuration error — check the Logs tab for the full traceback.',
      })
    }
  }

  if (error.includes('exit') && error.includes('code 1') && hints.length === 0) {
    hints.push({
      label: 'Process crashed',
      detail: 'The job exited with an error. Check the Logs tab for details, or ensure all dependencies (MLflow, GPU, AI Gateway) are configured.'
    })
  }

  if (hints.length === 0) return null

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Possible causes</p>
      {hints.map((hint, i) => (
        <div key={i} className="flex items-start gap-2.5 rounded-lg p-2 -mx-2 transition-colors duration-150 hover:bg-muted/40">
          <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-rh-yellow-light/50 text-rh-yellow dark:bg-rh-yellow-dark/40 dark:text-rh-yellow mt-px">
            <AlertCircle className="h-3 w-3" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium">{hint.label}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{hint.detail}</p>
            {hint.link && (
              <Link to={hint.link.to} className="inline-flex items-center gap-0.5 text-xs font-medium underline underline-offset-2 hover:text-foreground transition-colors mt-1">
                {hint.link.text} <ArrowRight className="h-2.5 w-2.5" />
              </Link>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function MetadataRow({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-2 py-2.5 -mx-2 rounded-lg transition-colors duration-150 hover:bg-muted/30 border-b border-border/30 last:border-0">
      <span className="text-sm text-muted-foreground shrink-0">{label}</span>
      <span className={`text-sm text-right ${mono ? "font-mono text-xs" : ""}`}>
        {value}
      </span>
    </div>
  )
}

function LogsTab({ job }: { job: Job }) {
  const isActive = ["running", "provisioning", "queued"].includes(job.status)
  const { data: lines, isLoading } = useJobLogs(job.id, isActive)

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        Loading logs...
      </div>
    )
  }

  if (!lines || lines.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        No logs available for this job.
      </div>
    )
  }

  return (
    <LogViewer lines={lines} isStreaming={isActive} className="flex-1 min-h-0" />
  )
}

function MetricsTab({ job }: { job: Job }) {
  const runId = job.type === "training" ? job.mlflow_run_id : null
  const { data, isLoading } = useJobMlflowMetrics(runId)

  if (job.type !== "training") {
    return (
      <div className="pt-4 text-sm text-muted-foreground text-center">
        Metrics are only available for training jobs.
      </div>
    )
  }

  if (!job.mlflow_run_id) {
    return (
      <div className="pt-4 text-sm text-muted-foreground text-center">
        No MLflow run associated with this job yet.
      </div>
    )
  }

  return (
    <div className="pt-2">
      <TrainingMetricsChart data={data} isLoading={isLoading} />
    </div>
  )
}

function ConfigTab({ job }: { job: Job }) {
  const [configOpen, setConfigOpen] = useState(true)
  const [metadataOpen, setMetadataOpen] = useState(false)

  return (
    <div className="space-y-2 pt-2">
      <Collapsible open={configOpen} onOpenChange={setConfigOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
          Config
          <ChevronDown
            className={`h-4 w-4 transition-transform duration-200 ${configOpen ? "rotate-180" : ""}`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
            <JsonTreeViewer data={job.config} collapsed={2} />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {job.metadata && Object.keys(job.metadata).length > 0 && (
        <Collapsible open={metadataOpen} onOpenChange={setMetadataOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
            Metadata
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${metadataOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
              <JsonTreeViewer data={job.metadata} collapsed={2} />
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}
