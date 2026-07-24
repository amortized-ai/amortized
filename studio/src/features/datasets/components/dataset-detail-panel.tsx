import { useState, useMemo } from "react"
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
import { EditableTitle } from "@/components/editable-title"
import { setMlflowRunTag } from "@/lib/api-client"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { JsonTreeViewer } from "@/components/json-tree-viewer"
import {
  Copy,
  X,
  Sparkles,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Search,
  ExternalLink,
  ArrowRight,
  Database,
  Thermometer,
  Bot,
  Calendar,
  Hash,
} from "lucide-react"
import { useDatasetSamples } from "../api/use-datasets"
import type { DatasetRecord, DatasetSample } from "@/types/api"

interface DatasetDetailPanelProps {
  dataset: DatasetRecord | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function DatasetDetailPanel({
  dataset,
  open,
  onOpenChange,
}: DatasetDetailPanelProps) {
  const queryClient = useQueryClient()
  if (!dataset) return null

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
                value={dataset.name}
                className="text-base font-semibold"
                onSave={async (newName) => {
                  await setMlflowRunTag(dataset.run_id, "dataset_name", newName)
                  queryClient.invalidateQueries({ queryKey: ["mlflow", "datasets"] })
                }}
              />
            </DialogTitle>
            <Badge
              variant="secondary"
              className="shrink-0 gap-1 bg-[#ece6ff] text-[#5e40be] border-0 dark:bg-[#1b0d33]/40 dark:text-[#876fd4]"
            >
              <Sparkles className="h-3 w-3" />
              SDG
            </Badge>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button variant="outline" size="sm" asChild>
              <a
                href={`/mlflow/#/experiments/${dataset.experiment_id}/runs/${dataset.run_id}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="mr-1 h-3.5 w-3.5" />
                MLflow
              </a>
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
                value="samples"
                className="flex-1 transition-all duration-200"
              >
                Samples
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
            <OverviewTab dataset={dataset} onClose={() => onOpenChange(false)} />
          </TabsContent>

          <TabsContent
            value="samples"
            className="mt-0 flex-1 min-h-0 flex flex-col px-6 py-4"
          >
            <SamplesTab dataset={dataset} />
          </TabsContent>

          <TabsContent
            value="config"
            className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
          >
            <ConfigTab dataset={dataset} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------------------

function OverviewTab({ dataset, onClose }: { dataset: DatasetRecord; onClose: () => void }) {
  const navigate = useNavigate()
  const [copied, setCopied] = useState(false)
  const sampleCount = dataset.metrics["num_samples_generated"] ?? null
  const model = dataset.params["model"] ?? null
  const temperature = dataset.params["temperature"] ?? null
  const jobId = dataset.tags["job_id"] ?? null

  function handleCopyId() {
    void navigator.clipboard.writeText(dataset.run_id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          icon={<Database className="h-4 w-4" />}
          label="Samples"
          value={sampleCount != null ? sampleCount.toLocaleString() : "--"}
          color="blue"
        />
        <StatCard
          icon={<Bot className="h-4 w-4" />}
          label="Teacher Model"
          value={model ?? "--"}
          color="purple"
          truncate
        />
        <StatCard
          icon={<Thermometer className="h-4 w-4" />}
          label="Temperature"
          value={temperature ?? "--"}
          color="orange"
        />
        <StatCard
          icon={<Calendar className="h-4 w-4" />}
          label="Created"
          value={new Date(dataset.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
          })}
          color="teal"
        />
      </div>

      {/* Metadata rows */}
      <div className="space-y-0">
        <MetadataRow
          label="Run ID"
          value={
            <span className="flex items-center gap-1.5">
              <Hash className="h-3 w-3 text-muted-foreground shrink-0" />
              <span className="font-mono text-xs">{dataset.run_id}</span>
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
          label="Experiment ID"
          value={
            <span className="font-mono text-xs">{dataset.experiment_id}</span>
          }
        />
        {dataset.run_name && (
          <MetadataRow label="Run Name" value={dataset.run_name} />
        )}
        <MetadataRow
          label="Created"
          value={new Date(dataset.created_at).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit",
          })}
        />
        {jobId && (
          <MetadataRow
            label="Source Job"
            value={
              <button
                type="button"
                onClick={() => {
                  onClose()
                  setTimeout(() => navigate(`/jobs?job=${encodeURIComponent(jobId)}`), 200)
                }}
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                {jobId}
                <ArrowRight className="h-3 w-3" />
              </button>
            }
          />
        )}
        {dataset.tags["mlflow.user"] && (
          <MetadataRow label="User" value={dataset.tags["mlflow.user"]} />
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

function contentToString(content: unknown): string {
  if (typeof content === "string") return content
  if (content == null) return ""
  return JSON.stringify(content, null, 2)
}

// ---------------------------------------------------------------------------
// Samples Tab
// ---------------------------------------------------------------------------

const SAMPLES_PER_PAGE = 10

function SamplesTab({ dataset }: { dataset: DatasetRecord }) {
  const { data: allSamples, isLoading, isError, error } = useDatasetSamples(
    dataset.experiment_id,
    dataset.run_id,
  )
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState("")

  const filtered = useMemo(() => {
    if (!allSamples) return []
    if (!search.trim()) return allSamples
    const q = search.toLowerCase()
    return allSamples.filter((s) =>
      s.messages.some((m) => contentToString(m.content).toLowerCase().includes(q)) ||
      Object.values(s.metadata).some(
        (v) => String(v).toLowerCase().includes(q),
      ),
    )
  }, [allSamples, search])

  const totalPages = Math.max(1, Math.ceil(filtered.length / SAMPLES_PER_PAGE))
  const pageSamples = filtered.slice(
    page * SAMPLES_PER_PAGE,
    (page + 1) * SAMPLES_PER_PAGE,
  )

  // Reset page when search changes
  const handleSearch = (value: string) => {
    setSearch(value)
    setPage(0)
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        Loading samples...
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
        <p>Could not load samples from this dataset.</p>
        <p className="text-xs font-mono">{(error as Error)?.message}</p>
        <p className="text-xs mt-1">
          The artifact file may not exist yet if the SDG job is still running.
        </p>
      </div>
    )
  }

  if (!allSamples || allSamples.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        No samples found in this dataset.
      </div>
    )
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-3">
      {/* Search and pagination header */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search sample content..."
            className="w-full rounded-md border bg-background px-8 py-1.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
          {search && (
            <button
              onClick={() => handleSearch("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <span className="text-xs text-muted-foreground shrink-0">
          {filtered.length} sample{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Sample list */}
      <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1">
        {pageSamples.map((sample) => (
          <SampleCard key={sample.index} sample={sample} />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between shrink-0 pt-1 border-t">
          <Button
            variant="ghost"
            size="sm"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Previous
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {page + 1} of {totalPages}
          </span>
          <Button
            variant="ghost"
            size="sm"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      )}
    </div>
  )
}

function SampleCard({ sample }: { sample: DatasetSample }) {
  const [expanded, setExpanded] = useState(false)
  const metaEntries = Object.entries(sample.metadata).filter(
    ([key]) => !key.startsWith("_"),
  )
  const hasMessages = sample.messages.length > 0

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      {/* Sample header */}
      <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b">
        <span className="text-xs font-medium text-muted-foreground">
          Sample #{sample.index + 1}
        </span>
        <span className="text-xs text-muted-foreground">
          {hasMessages
            ? `${sample.messages.length} message${sample.messages.length !== 1 ? "s" : ""}`
            : `${metaEntries.length} field${metaEntries.length !== 1 ? "s" : ""}`}
        </span>
      </div>

      {/* Messages (when present) */}
      {hasMessages && (
        <div className="p-3 space-y-2">
          {sample.messages.map((msg, i) => (
            <MessageBubble key={i} role={msg.role} content={msg.content} />
          ))}
        </div>
      )}

      {/* Flat data display (when no messages) */}
      {!hasMessages && metaEntries.length > 0 && (
        <div className="p-3 space-y-1.5">
          {metaEntries.map(([key, value]) => (
            <div key={key} className="flex items-baseline gap-2">
              <span className="text-xs font-medium text-muted-foreground shrink-0">{key}</span>
              <span className="text-sm break-words">{String(value)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Metadata toggle (only when messages are present — flat data shown inline above) */}
      {hasMessages && metaEntries.length > 0 && (
        <div className="px-3 pb-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronDown
              className={`h-3 w-3 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
            />
            {metaEntries.length} metadata field{metaEntries.length !== 1 ? "s" : ""}
          </button>
          {expanded && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {metaEntries.map(([key, value]) => (
                <Badge
                  key={key}
                  variant="outline"
                  className="font-mono text-xs py-0"
                >
                  {key}: {String(value)}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MessageBubble({ role, content }: { role: string; content: unknown }) {
  const text = contentToString(content)
  const isSystem = role === "system"
  const isAssistant = role === "assistant"

  const roleColors: Record<string, string> = {
    system:
      "bg-muted/50 text-muted-foreground border border-border/50",
    user: "bg-[#e0f0ff] text-foreground dark:bg-[#003366]/60",
    assistant:
      "bg-[#ece6ff] text-foreground dark:bg-[#1b0d33]/60",
  }

  const roleLabelColors: Record<string, string> = {
    system: "text-muted-foreground",
    user: "text-[#0066cc] dark:text-[#4394e5]",
    assistant: "text-[#5e40be] dark:text-[#876fd4]",
  }

  return (
    <div
      className={`flex flex-col ${isAssistant ? "items-end" : "items-start"}`}
    >
      <span
        className={`text-[10px] font-medium uppercase tracking-wider mb-0.5 px-1 ${roleLabelColors[role] ?? "text-muted-foreground"}`}
      >
        {role}
      </span>
      <div
        className={`rounded-lg px-3 py-2 text-sm leading-relaxed max-w-[85%] ${roleColors[role] ?? "bg-muted"}`}
      >
        <p
          className={`whitespace-pre-wrap break-words ${isSystem ? "text-xs" : "text-sm"}`}
          style={{ fontFamily: isSystem ? "var(--font-mono)" : undefined }}
        >
          {text}
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Config Tab
// ---------------------------------------------------------------------------

function ConfigTab({ dataset }: { dataset: DatasetRecord }) {
  const [paramsOpen, setParamsOpen] = useState(true)
  const [tagsOpen, setTagsOpen] = useState(false)
  const [metricsOpen, setMetricsOpen] = useState(false)

  return (
    <div className="space-y-2 pt-2">
      {/* Parameters */}
      <Collapsible open={paramsOpen} onOpenChange={setParamsOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
          Parameters
          <ChevronDown
            className={`h-4 w-4 transition-transform duration-200 ${paramsOpen ? "rotate-180" : ""}`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
            {Object.keys(dataset.params).length > 0 ? (
              <JsonTreeViewer data={dataset.params} collapsed={1} />
            ) : (
              <p className="text-sm text-muted-foreground">
                No parameters recorded.
              </p>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Metrics */}
      {Object.keys(dataset.metrics).length > 0 && (
        <Collapsible open={metricsOpen} onOpenChange={setMetricsOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
            Metrics
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${metricsOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
              <dl className="space-y-2 text-sm">
                {Object.entries(dataset.metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <dt className="text-muted-foreground">{key}</dt>
                    <dd className="font-mono">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Tags */}
      <Collapsible open={tagsOpen} onOpenChange={setTagsOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium hover:bg-muted transition-colors duration-150">
          Tags ({Object.keys(dataset.tags).length})
          <ChevronDown
            className={`h-4 w-4 transition-transform duration-200 ${tagsOpen ? "rotate-180" : ""}`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-1 rounded-lg bg-muted/50 p-4 overflow-x-auto">
            <div className="flex flex-wrap gap-2">
              {Object.entries(dataset.tags).map(([key, value]) => (
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
    </div>
  )
}
