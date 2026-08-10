import { useState, useMemo, useCallback } from "react"
import { Link, useNavigate } from "react-router"
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge as FlowEdge,
  type NodeMouseHandler,
} from "@xyflow/react"
import { Graph, layout } from "@dagrejs/dagre"
import "@xyflow/react/dist/style.css"
import { Workflow, ArrowRight, MessageSquare, BookOpen } from "lucide-react"
import { PageHeader } from "@/components/page-header"
import { SearchInput } from "@/components/search-input"
import { FilterChips, STATUS_COLORS } from "@/features/jobs/components/filter-chips"
import { StatusBadge } from "@/features/jobs/components/status-badge"
import { ErrorState } from "@/components/error-state"
import { TableSkeleton } from "@/components/table-skeleton"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "@/components/ui/empty"
import { Button } from "@/components/ui/button"
import { useLineageChains } from "./api/use-lineage"
import { LineageNode, type LineageFlowNode } from "@/features/jobs/components/lineage-node"
import type { LineageChainSummary, LineageResponse, JobStatus } from "@/types/api"

const NODE_WIDTH = 220
const NODE_HEIGHT = 130

const nodeTypes = { lineage: LineageNode } as const

const STATUS_OPTIONS: { value: JobStatus; label: string }[] = [
  { value: "succeeded", label: "Succeeded", color: STATUS_COLORS.succeeded },
  { value: "running", label: "Running", color: STATUS_COLORS.running },
  { value: "failed", label: "Failed", color: STATUS_COLORS.failed },
  { value: "queued", label: "Queued", color: STATUS_COLORS.queued },
  { value: "provisioning", label: "Provisioning", color: STATUS_COLORS.provisioning },
  { value: "cancelled", label: "Cancelled", color: STATUS_COLORS.cancelled },
] as { value: JobStatus; label: string; color: (typeof STATUS_COLORS)[string] }[]

function buildLayoutedGraph(data: LineageResponse) {
  const g = new Graph({ directed: true })
  g.setGraph({ rankdir: "LR", nodesep: 40, ranksep: 80 })
  g.setDefaultEdgeLabel(() => ({}))

  for (const node of data.nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const edge of data.edges) {
    g.setEdge(edge.source, edge.target)
  }

  layout(g)

  const nodes: LineageFlowNode[] = data.nodes.map((node) => {
    const pos = g.node(node.id)
    return {
      id: node.id,
      type: "lineage" as const,
      position: {
        x: (pos?.x ?? 0) - NODE_WIDTH / 2,
        y: (pos?.y ?? 0) - NODE_HEIGHT / 2,
      },
      data: {
        nodeType: node.type,
        status: node.status,
        recipe: node.recipe,
        meta: node.meta,
        isTarget: node.id === data.target_id,
      },
    }
  })

  const edges: FlowEdge[] = data.edges.map((edge) => ({
    id: `${edge.source}->${edge.target}`,
    source: edge.source,
    target: edge.target,
    animated: false,
    style: { stroke: "var(--color-border)", strokeWidth: 2 },
  }))

  return { nodes, edges }
}

function ChainGraph({ data }: { data: LineageResponse }) {
  const { fitView } = useReactFlow()
  const navigate = useNavigate()

  const { nodes, edges } = useMemo(() => buildLayoutedGraph(data), [data])

  const onInit = useCallback(() => {
    setTimeout(() => fitView({ padding: 0.2, duration: 200 }), 50)
  }, [fitView])

  const onNodeClick: NodeMouseHandler<LineageFlowNode> = useCallback(
    (_event, node) => {
      void navigate(`/jobs?job=${node.id}`)
    },
    [navigate],
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
      panOnDrag={false}
      zoomOnScroll={false}
      zoomOnPinch={false}
      zoomOnDoubleClick={false}
      preventScrolling={false}
      proOptions={{ hideAttribution: true }}
      minZoom={0.3}
      maxZoom={1}
      className="cursor-default"
    />
  )
}

function formatRelativeTime(dateStr: string): string {
  const now = Date.now()
  const then = new Date(dateStr).getTime()
  const diff = now - then
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return "just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function ChainCard({ chain }: { chain: LineageChainSummary }) {
  const nodeCount = chain.lineage.nodes.length
  const graphHeight = nodeCount <= 2 ? 160 : nodeCount <= 4 ? 200 : 240

  return (
    <div className="rounded-lg border bg-card shadow-sm transition-shadow hover:shadow-md">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
            <Workflow className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold">{chain.name}</h3>
            <p className="text-xs text-muted-foreground">
              {chain.job_count} {chain.job_count === 1 ? "job" : "jobs"}
              {" · "}
              started {formatRelativeTime(chain.created_at)}
              {" · "}
              updated {formatRelativeTime(chain.updated_at)}
            </p>
          </div>
        </div>
        <StatusBadge status={chain.latest_status} />
      </div>
      <div style={{ height: graphHeight }}>
        <ReactFlowProvider>
          <ChainGraph data={chain.lineage} />
        </ReactFlowProvider>
      </div>
    </div>
  )
}

export default function LineagePage() {
  const { data: chains = [], isLoading, isError, error, refetch } = useLineageChains()
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<JobStatus[]>([])

  const filteredChains = useMemo(() => {
    let result = chains
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter((c) => c.name.toLowerCase().includes(q))
    }
    if (statusFilter.length > 0) {
      result = result.filter((c) => statusFilter.includes(c.latest_status))
    }
    return result
  }, [chains, search, statusFilter])

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Lineage"
        description="End-to-end experiment tracking across your pipelines"
      />

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <SearchInput
          value={search}
          onChange={setSearch}
          placeholder="Search pipelines..."
        />
        <FilterChips
          label="Status"
          options={STATUS_OPTIONS}
          selected={statusFilter}
          onChange={setStatusFilter}
        />
      </div>

      {isLoading ? (
        <TableSkeleton columns={3} />
      ) : filteredChains.length > 0 ? (
        <div className="space-y-4">
          {filteredChains.map((chain) => (
            <ChainCard key={chain.chain_id} chain={chain} />
          ))}
        </div>
      ) : chains.length > 0 ? (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Workflow /></EmptyMedia>
            <EmptyTitle>No matching pipelines</EmptyTitle>
            <EmptyDescription>Try adjusting your search or filters.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Workflow /></EmptyMedia>
            <EmptyTitle>No experiment chains found</EmptyTitle>
            <EmptyDescription>
              Pipelines appear here when you chain jobs together (e.g. data generation followed by training).
            </EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <div className="flex gap-2">
              <Button asChild size="sm">
                <Link to="/chat">
                  <MessageSquare className="mr-1.5 h-3.5 w-3.5" />
                  Chat
                  <ArrowRight className="ml-1 h-3 w-3" />
                </Link>
              </Button>
              <Button asChild size="sm" variant="outline">
                <Link to="/recipes">
                  <BookOpen className="mr-1.5 h-3.5 w-3.5" />
                  Recipes
                </Link>
              </Button>
            </div>
          </EmptyContent>
        </Empty>
      )}
    </div>
  )
}
