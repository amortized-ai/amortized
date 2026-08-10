import { useState, useMemo, useCallback } from "react"
import { useNavigate } from "react-router"
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  type NodeMouseHandler,
} from "@xyflow/react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { StatusBadge } from "@/features/jobs/components/status-badge"
import {
  nodeTypes,
  buildLayoutedGraph,
} from "@/features/jobs/utils/layout-graph"
import type { LineageFlowNode } from "@/features/jobs/components/lineage-node"
import type { LineageChainSummary, JobStatus } from "@/types/api"
import {
  X,
  Sparkles,
  GraduationCap,
  ArrowRight,
  Calendar,
  Database,
  Box,
} from "lucide-react"
import { useDatasets } from "@/features/datasets/api/use-datasets"

interface ExperimentDetailPanelProps {
  chain: LineageChainSummary | null
  open: boolean
  onOpenChange: (open: boolean) => void
  displayName: string
  onRename: (id: string, name: string) => void
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

function ExperimentGraph({
  chain,
  onClose,
}: {
  chain: LineageChainSummary
  onClose: () => void
}) {
  const { fitView } = useReactFlow()
  const navigate = useNavigate()

  const { nodes, edges } = useMemo(
    () => buildLayoutedGraph(chain.lineage, { nodesep: 60, ranksep: 100 }),
    [chain.lineage],
  )

  const onInit = useCallback(() => {
    setTimeout(() => fitView({ padding: 0.15, duration: 300 }), 50)
  }, [fitView])

  const onNodeClick: NodeMouseHandler<LineageFlowNode> = useCallback(
    (_event, node) => {
      const nodeType = node.data.nodeType
      onClose()

      setTimeout(() => {
        if (nodeType === "dataset") {
          const mlflowRunId = (node.data.meta as Record<string, unknown>)?.mlflow_run_id
          if (typeof mlflowRunId === "string" && mlflowRunId) {
            void navigate(`/datasets?dataset=${encodeURIComponent(mlflowRunId)}`)
          } else {
            const link = node.data.link
            if (link) {
              const runId = link.split("/").pop()
              void navigate(`/datasets?dataset=${encodeURIComponent(runId ?? "")}`)
            }
          }
        } else if (nodeType === "model") {
          const sourceEdge = chain.lineage.edges.find((e) => e.target === node.id)
          if (sourceEdge) {
            void navigate(`/jobs?job=${encodeURIComponent(sourceEdge.source)}`)
          }
        } else {
          const link = node.data.link
          if (link) void navigate(link)
        }
      }, 200)
    },
    [navigate, onClose, chain.lineage.edges],
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onInit={onInit}
      onNodeClick={onNodeClick}
      fitView
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      panOnScroll
      proOptions={{ hideAttribution: true }}
      minZoom={0.3}
      maxZoom={1.5}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} className="!bg-background" />
      <Controls
        showInteractive={false}
        className="!border-border !bg-card !shadow-sm [&>button]:!border-border [&>button]:!bg-card [&>button]:!fill-foreground"
      />
    </ReactFlow>
  )
}

function JobsTab({
  chain,
  onClose,
}: {
  chain: LineageChainSummary
  onClose: () => void
}) {
  const navigate = useNavigate()
  const jobNodes = chain.lineage.nodes.filter((n) =>
    ["sdg", "training", "upload", "eval"].includes(n.type),
  )

  return (
    <div className="space-y-2">
      {jobNodes.map((node) => {
        const isTraining = node.type === "training"
        return (
          <button
            key={node.id}
            onClick={() => {
              onClose()
              setTimeout(() => void navigate(`/jobs?job=${encodeURIComponent(node.id)}`), 200)
            }}
            className="w-full flex items-center gap-3 rounded-lg border bg-card px-4 py-3 text-left transition-all hover:shadow-sm hover:ring-2 hover:ring-primary/20"
          >
            <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${
              isTraining
                ? "bg-[#e0f0ff] dark:bg-[#003366]/40"
                : "bg-[#ece6ff] dark:bg-[#1b0d33]/40"
            }`}>
              {isTraining ? (
                <GraduationCap className={`h-4 w-4 ${
                  isTraining ? "text-[#0066cc] dark:text-[#4394e5]" : ""
                }`} />
              ) : (
                <Sparkles className="h-4 w-4 text-[#5e40be] dark:text-[#876fd4]" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">
                  {node.type === "sdg" ? "Data Generation" : node.type.charAt(0).toUpperCase() + node.type.slice(1)}
                </span>
                <StatusBadge status={node.status as JobStatus} />
              </div>
              <div className="flex items-center gap-3 mt-0.5">
                {node.meta?.model && (
                  <span className="text-xs text-muted-foreground truncate">
                    {String(node.meta.model)}
                  </span>
                )}
                {node.meta?.algorithm && (
                  <span className="text-xs text-muted-foreground">
                    {String(node.meta.algorithm)}
                  </span>
                )}
                {node.meta?.num_records && (
                  <span className="text-xs text-muted-foreground">
                    {Number(node.meta.num_records).toLocaleString()} records
                  </span>
                )}
                {node.created_at && (
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {formatDate(node.created_at)}
                  </span>
                )}
              </div>
            </div>
            <ArrowRight className="h-4 w-4 text-muted-foreground/40 shrink-0" />
          </button>
        )
      })}
    </div>
  )
}

function ArtifactsTab({
  chain,
  onClose,
}: {
  chain: LineageChainSummary
  onClose: () => void
}) {
  const navigate = useNavigate()
  const { data: datasets = [] } = useDatasets()
  const artifactNodes = chain.lineage.nodes.filter((n) =>
    ["dataset", "model"].includes(n.type),
  )

  const datasetNameMap = useMemo(() => {
    const map: Record<string, string> = {}
    for (const d of datasets) {
      map[d.run_id] = d.name
    }
    return map
  }, [datasets])

  function resolveDatasetName(node: typeof artifactNodes[number]): string {
    const runId = (node.meta as Record<string, unknown>)?.mlflow_run_id
    if (typeof runId === "string" && datasetNameMap[runId]) return datasetNameMap[runId]
    const name = (node.meta as Record<string, unknown>)?.name
    if (typeof name === "string" && name !== "Generated dataset") return name
    return "Dataset"
  }

  return (
    <div className="space-y-2">
      {artifactNodes.map((node) => {
        const isModel = node.type === "model"
        const displayName = isModel
          ? ((node.meta?.name as string) || "Model")
          : resolveDatasetName(node)
        return (
          <button
            key={node.id}
            onClick={() => {
              onClose()
              setTimeout(() => {
                if (isModel) {
                  const sourceEdge = chain.lineage.edges.find((e) => e.target === node.id)
                  if (sourceEdge) {
                    void navigate(`/jobs?job=${encodeURIComponent(sourceEdge.source)}`)
                  }
                } else {
                  const runId = (node.meta as Record<string, unknown>)?.mlflow_run_id
                  if (typeof runId === "string" && runId) {
                    void navigate(`/datasets?dataset=${encodeURIComponent(runId)}`)
                  } else if (node.link) {
                    const id = node.link.split("/").pop()
                    void navigate(`/datasets?dataset=${encodeURIComponent(id ?? "")}`)
                  }
                }
              }, 200)
            }}
            className="w-full flex items-center gap-3 rounded-lg border bg-card px-4 py-3 text-left transition-all hover:shadow-sm hover:ring-2 hover:ring-primary/20"
          >
            <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${
              isModel
                ? "bg-[#e0f0ff] dark:bg-[#003366]/40"
                : "bg-[#ece6ff] dark:bg-[#1b0d33]/40"
            }`}>
              {isModel ? (
                <Box className="h-4 w-4 text-[#0066cc] dark:text-[#4394e5]" />
              ) : (
                <Database className="h-4 w-4 text-[#5e40be] dark:text-[#876fd4]" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium truncate">
                  {displayName}
                </span>
              </div>
              <div className="flex items-center gap-3 mt-0.5">
                {!isModel && node.meta?.num_records && (
                  <span className="text-xs text-muted-foreground">
                    {Number(node.meta.num_records).toLocaleString()} records
                  </span>
                )}
                <span className="text-xs text-muted-foreground">
                  {isModel ? "View training job" : "View dataset"} →
                </span>
              </div>
            </div>
            <ArrowRight className="h-4 w-4 text-muted-foreground/40 shrink-0" />
          </button>
        )
      })}
    </div>
  )
}

export function ExperimentDetailPanel({
  chain,
  open,
  onOpenChange,
  displayName,
}: ExperimentDetailPanelProps) {
  const [editingName, setEditingName] = useState(false)
  const [nameValue, setNameValue] = useState(displayName)

  const jobCount = useMemo(
    () =>
      chain
        ? chain.lineage.nodes.filter((n) =>
            ["sdg", "training", "upload", "eval"].includes(n.type),
          ).length
        : 0,
    [chain],
  )

  const artifactCount = useMemo(
    () =>
      chain
        ? chain.lineage.nodes.filter((n) =>
            ["dataset", "model"].includes(n.type),
          ).length
        : 0,
    [chain],
  )

  if (!chain) return null

  const handleClose = () => onOpenChange(false)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        variant="panel"
        showCloseButton={false}
        className="!max-w-[1000px] !w-[1000px] !h-[80vh] flex flex-col !gap-0 p-0 overflow-hidden rounded-xl"
      >
        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <DialogTitle asChild>
              {editingName ? (
                <input
                  autoFocus
                  value={nameValue}
                  onChange={(e) => setNameValue(e.target.value)}
                  onBlur={() => setEditingName(false)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setEditingName(false)
                    if (e.key === "Escape") {
                      setNameValue(displayName)
                      setEditingName(false)
                    }
                  }}
                  className="text-base font-semibold bg-transparent border-b border-primary/40 outline-none"
                />
              ) : (
                <h2
                  className="text-base font-semibold cursor-text hover:text-primary/80 transition-colors"
                  onDoubleClick={() => {
                    setNameValue(displayName)
                    setEditingName(true)
                  }}
                >
                  {displayName}
                </h2>
              )}
            </DialogTitle>
            <StatusBadge status={chain.latest_status} />
            <span className="text-xs text-muted-foreground">
              {jobCount} {jobCount === 1 ? "job" : "jobs"}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 opacity-70 hover:opacity-100"
            onClick={handleClose}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Close</span>
          </Button>
        </div>

        <Tabs defaultValue="graph" className="flex flex-col flex-1 min-h-0">
          <div className="px-6 pt-3 shrink-0">
            <TabsList className="w-full">
              <TabsTrigger value="graph" className="flex-1 transition-all duration-200">
                Pipeline
              </TabsTrigger>
              <TabsTrigger value="jobs" className="flex-1 transition-all duration-200">
                Jobs ({jobCount})
              </TabsTrigger>
              <TabsTrigger value="artifacts" className="flex-1 transition-all duration-200">
                Artifacts ({artifactCount})
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="graph" className="mt-0 flex-1 min-h-0">
            <ReactFlowProvider>
              <ExperimentGraph chain={chain} onClose={handleClose} />
            </ReactFlowProvider>
          </TabsContent>

          <TabsContent value="jobs" className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4">
            <JobsTab chain={chain} onClose={handleClose} />
          </TabsContent>

          <TabsContent value="artifacts" className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4">
            <ArtifactsTab chain={chain} onClose={handleClose} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
