import { useState } from "react"
import { useNavigate } from "react-router"
import { useQueryClient } from "@tanstack/react-query"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { EditableTitle } from "@/components/editable-title"
import { JsonTreeViewer } from "@/components/json-tree-viewer"
import { setMlflowRegisteredModelTag } from "@/lib/api-client"
import { DeleteEntityDialog } from "@/components/delete-entity-dialog"
import { TrainingMetricsChart } from "@/features/jobs/components/training-metrics-chart"
import { useModelRunData, useModelJobs, useDeleteModel } from "../api/use-models"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import { formatDuration } from "../lib/format"
import type { ModelRecord } from "@/types/api"
import type { AllMetricsData } from "@/features/jobs/api/use-jobs"
import {
  Copy,
  X,
  ExternalLink,
  ArrowRight,
  Trash2,
  GraduationCap,
  Calendar,
  Clock,
  Cpu,
  Hash,
  ChevronDown,
} from "lucide-react"

interface ModelDetailPanelProps {
  model: ModelRecord | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function ModelDetailPanel({
  model,
  open,
  onOpenChange,
}: ModelDetailPanelProps) {
  const queryClient = useQueryClient()
  const deleteMutation = useDeleteModel()
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const getName = useEntityNamesStore((s) => s.getName)
  const { data: runData } = useModelRunData(model?.run_id || null)

  if (!model) return null

  const displayName =
    runData?.tags?.["model_display_name"] ||
    model.tags?.model_display_name ||
    getName(model.name) ||
    model.name

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        variant="panel"
        showCloseButton={false}
        className="!max-w-[900px] !w-[900px] !h-[80vh] flex flex-col !gap-0 p-0 overflow-hidden rounded-xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <DialogTitle asChild>
              <EditableTitle
                value={displayName}
                className="text-base font-semibold"
                onSave={async (newName) => {
                  await setMlflowRegisteredModelTag(model.name, "model_display_name", newName)
                  queryClient.invalidateQueries({ queryKey: ["mlflow", "models"] })
                }}
              />
            </DialogTitle>
            {model.aliases.map((alias) => (
              <Badge
                key={alias}
                variant="secondary"
                className="shrink-0 gap-1 font-mono text-xs"
              >
                @{alias}
              </Badge>
            ))}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button variant="outline" size="sm" asChild>
              <a
                href={`/mlflow/#/models/${encodeURIComponent(model.name)}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="mr-1 h-3.5 w-3.5" />
                MLflow
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
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

        {/* Tabs */}
        <Tabs defaultValue="overview" className="flex flex-col flex-1 min-h-0">
          <div className="px-6 pt-3 shrink-0">
            <TabsList className="w-full">
              <TabsTrigger
                value="overview"
                className="flex-1 transition-all duration-200"
              >
                Overview
              </TabsTrigger>
              <TabsTrigger
                value="metrics"
                className="flex-1 transition-all duration-200"
              >
                Metrics
              </TabsTrigger>
              <TabsTrigger
                value="config"
                className="flex-1 transition-all duration-200"
              >
                Config
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent
            value="overview"
            className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
          >
            <OverviewTab model={model} onClose={() => onOpenChange(false)} />
          </TabsContent>

          <TabsContent
            value="metrics"
            className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
          >
            <MetricsTab runId={model.run_id} />
          </TabsContent>

          <TabsContent
            value="config"
            className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
          >
            <ConfigTab runId={model.run_id} />
          </TabsContent>
        </Tabs>

        <DeleteEntityDialog
          open={deleteDialogOpen}
          entityType="model"
          entityName={displayName}
          onConfirm={() => {
            deleteMutation.mutate(model.name, {
              onSuccess: () => {
                setDeleteDialogOpen(false)
                onOpenChange(false)
              },
            })
          }}
          onCancel={() => setDeleteDialogOpen(false)}
          isPending={deleteMutation.isPending}
        />
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------------------

function OverviewTab({ model, onClose }: { model: ModelRecord; onClose: () => void }) {
  const navigate = useNavigate()
  const { data: runData } = useModelRunData(model.run_id || null)
  const { data: jobsData } = useModelJobs(model.run_id || null)
  const [copied, setCopied] = useState(false)

  const algorithm = String(
    jobsData?.trainingJob?.config?.algorithm ??
    (runData?.params["osft"] === "True" ? "osft" : null) ??
    runData?.params["algorithm"] ??
    "--"
  )
  const baseModel = String(
    runData?.params["model_name_or_path"] ??
    runData?.params["model_path"] ??
    jobsData?.trainingJob?.config?.model_name_or_path ??
    "--"
  )
  const finalLoss = runData?.finalMetrics["loss"] ?? runData?.finalMetrics["avg_loss_backward"]
  const epochs = runData?.finalMetrics["epoch"] ?? (runData?.params["max_epochs"] ? Number(runData.params["max_epochs"]) : undefined)
  const trainingJob = jobsData?.trainingJob
  const trainDuration = trainingJob?.started_at && trainingJob?.completed_at
    ? (new Date(trainingJob.completed_at).getTime() - new Date(trainingJob.started_at).getTime()) / 1000
    : undefined

  function handleCopyId() {
    void navigator.clipboard.writeText(model.run_id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          icon={<GraduationCap className="h-4 w-4" />}
          label="Final Loss"
          value={finalLoss != null ? finalLoss.toFixed(4) : "--"}
          color="blue"
        />
        <StatCard
          icon={<Cpu className="h-4 w-4" />}
          label="Algorithm"
          value={algorithm.toUpperCase()}
          color="purple"
        />
        <StatCard
          icon={<Clock className="h-4 w-4" />}
          label="Training Time"
          value={trainDuration != null ? formatDuration(trainDuration) : "--"}
          color="orange"
        />
        <StatCard
          icon={<Calendar className="h-4 w-4" />}
          label="Created"
          value={new Date(model.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
          })}
          color="teal"
        />
      </div>

      {/* Metadata rows */}
      <div className="space-y-0">
        <MetadataRow label="Base Model" value={baseModel} />
        <MetadataRow
          label="Epochs"
          value={epochs != null ? String(Math.round(epochs)) : "--"}
        />
        <MetadataRow label="Version" value={`v${model.version}`} />
        <MetadataRow
          label="Run ID"
          value={
            <span className="flex items-center gap-1.5">
              <Hash className="h-3 w-3 text-muted-foreground shrink-0" />
              <span className="font-mono text-xs">{model.run_id}</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 w-5 p-0 shrink-0"
                onClick={handleCopyId}
                aria-label="Copy run ID"
              >
                <Copy className="h-3 w-3" />
              </Button>
              {copied && (
                <span className="text-xs text-muted-foreground">Copied!</span>
              )}
            </span>
          }
        />
        <MetadataRow
          label="Source"
          value={
            <span className="font-mono text-xs truncate max-w-[400px]" title={model.source}>
              {model.source}
            </span>
          }
        />
        {jobsData?.trainingJob && (
          <MetadataRow
            label="Training Job"
            value={
              <button
                type="button"
                onClick={() => {
                  onClose()
                  setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(jobsData.trainingJob!.id)}`), 200)
                }}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                {jobsData.trainingJob.id.slice(0, 8)}
                <ArrowRight className="h-3 w-3" />
              </button>
            }
          />
        )}
        {jobsData?.sdgJob && (
          <MetadataRow
            label="SDG Job"
            value={
              <button
                type="button"
                onClick={() => {
                  onClose()
                  setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(jobsData.sdgJob!.id)}`), 200)
                }}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                {jobsData.sdgJob.id.slice(0, 8)}
                <ArrowRight className="h-3 w-3" />
              </button>
            }
          />
        )}
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  color,
  truncate,
}: {
  icon: React.ReactNode
  label: string
  value: string | number
  color: "blue" | "purple" | "orange" | "teal"
  truncate?: boolean
}) {
  const colorMap = {
    blue: "bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]",
    purple:
      "bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]",
    orange:
      "bg-[#fff4cc] text-[#b98412] dark:bg-[#54330b]/40 dark:text-[#dca614]",
    teal: "bg-[#daf2f2] text-[#147878] dark:bg-[#003333]/40 dark:text-[#37a3a3]",
  }

  return (
    <div className="rounded-lg border bg-card p-3 space-y-1.5">
      <div className="flex items-center gap-2">
        <div
          className={`flex h-6 w-6 items-center justify-center rounded-md ${colorMap[color]}`}
        >
          {icon}
        </div>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p
        className={`text-sm font-semibold ${truncate ? "truncate" : ""}`}
        title={typeof value === "string" ? value : undefined}
      >
        {value}
      </p>
    </div>
  )
}

function MetadataRow({
  label,
  value,
}: {
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-2 py-2.5 -mx-2 rounded-lg transition-colors duration-150 hover:bg-muted/30 border-b border-border/30 last:border-0">
      <span className="text-sm text-muted-foreground shrink-0">{label}</span>
      <span className="text-sm text-right">{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Metrics Tab
// ---------------------------------------------------------------------------

function MetricsTab({ runId }: { runId: string }) {
  const { data: runData, isLoading } = useModelRunData(runId || null)
  const { data: jobsData } = useModelJobs(runId || null)

  const metricsData: AllMetricsData | undefined = runData
    ? {
        histories: runData.histories,
        finalValues: { ...runData.finalMetrics },
        params: runData.params,
      }
    : undefined

  if (metricsData && !metricsData.finalValues["train_runtime"]) {
    const job = jobsData?.trainingJob
    if (job?.started_at && job?.completed_at) {
      metricsData.finalValues["train_runtime"] =
        (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000
    }
  }

  return <TrainingMetricsChart data={metricsData} isLoading={isLoading} />
}

// ---------------------------------------------------------------------------
// Config Tab
// ---------------------------------------------------------------------------

function ConfigTab({ runId }: { runId: string }) {
  const { data: runData } = useModelRunData(runId || null)
  const { data: jobsData } = useModelJobs(runId || null)
  const [paramsOpen, setParamsOpen] = useState(true)
  const [tagsOpen, setTagsOpen] = useState(false)
  const [jobConfigOpen, setJobConfigOpen] = useState(false)

  const hasParams = runData && Object.keys(runData.params).length > 0
  const hasTags = runData && Object.keys(runData.tags).length > 0
  const jobConfig = jobsData?.trainingJob?.config

  return (
    <div className="space-y-2 pt-2">
      {/* Training Parameters */}
      {hasParams && (
        <Collapsible open={paramsOpen} onOpenChange={setParamsOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
            Training Parameters
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${paramsOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
              <JsonTreeViewer data={runData!.params} collapsed={1} />
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Job Config */}
      {jobConfig && Object.keys(jobConfig).length > 0 && (
        <Collapsible open={jobConfigOpen} onOpenChange={setJobConfigOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
            Job Config
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${jobConfigOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
              <JsonTreeViewer data={jobConfig} collapsed={2} />
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Tags */}
      {hasTags && (
        <Collapsible open={tagsOpen} onOpenChange={setTagsOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
            Tags ({Object.keys(runData!.tags).length})
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${tagsOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
              <div className="flex flex-wrap gap-2">
                {Object.entries(runData!.tags)
                  .filter(([key]) => !key.startsWith("mlflow."))
                  .map(([key, value]) => (
                    <Badge
                      key={key}
                      variant="outline"
                      className="font-mono text-xs"
                    >
                      {key}: {value}
                    </Badge>
                  ))}
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}
