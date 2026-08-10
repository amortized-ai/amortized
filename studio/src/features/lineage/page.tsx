import { useState, useMemo, useCallback, useRef, useEffect } from "react"
import { Link } from "react-router"
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react"
import {
  Workflow,
  ArrowRight,
  MessageSquare,
  BookOpen,
  Sparkles,
  GraduationCap,
  Pencil,
} from "lucide-react"
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
import { ExperimentDetailPanel } from "./components/experiment-detail-panel"
import {
  nodeTypes,
  buildLayoutedGraph,
} from "@/features/jobs/utils/layout-graph"
import type { LineageChainSummary, LineageResponse, JobStatus } from "@/types/api"

const NAMES_STORAGE_KEY = "amortized:experiment-names"

function loadExperimentNames(): Record<string, string> {
  try {
    const raw = localStorage.getItem(NAMES_STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveExperimentName(chainId: string, name: string) {
  const names = loadExperimentNames()
  names[chainId] = name
  localStorage.setItem(NAMES_STORAGE_KEY, JSON.stringify(names))
}

const STATUS_OPTIONS: { value: JobStatus; label: string }[] = [
  { value: "succeeded", label: "Succeeded", color: STATUS_COLORS.succeeded },
  { value: "running", label: "Running", color: STATUS_COLORS.running },
  { value: "failed", label: "Failed", color: STATUS_COLORS.failed },
  { value: "queued", label: "Queued", color: STATUS_COLORS.queued },
  { value: "provisioning", label: "Provisioning", color: STATUS_COLORS.provisioning },
  { value: "cancelled", label: "Cancelled", color: STATUS_COLORS.cancelled },
] as { value: JobStatus; label: string; color: (typeof STATUS_COLORS)[string] }[]

function ChainGraphPreview({ data }: { data: LineageResponse }) {
  const { nodes, edges } = useMemo(
    () => buildLayoutedGraph(data, { nodesep: 40, ranksep: 70 }),
    [data],
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.1 }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      panOnDrag={false}
      panOnScroll={false}
      zoomOnScroll={false}
      zoomOnPinch={false}
      zoomOnDoubleClick={false}
      preventScrolling={false}
      proOptions={{ hideAttribution: true }}
      minZoom={0.1}
      maxZoom={1}
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

function EditableTitle({
  chainId,
  defaultName,
  customNames,
  onRename,
}: {
  chainId: string
  defaultName: string
  customNames: Record<string, string>
  onRename: (id: string, name: string) => void
}) {
  const displayName = customNames[chainId] || defaultName
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState(displayName)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [editing])

  const commit = useCallback(() => {
    const trimmed = editValue.trim()
    if (trimmed && trimmed !== defaultName) {
      onRename(chainId, trimmed)
    }
    setEditing(false)
  }, [editValue, chainId, defaultName, onRename])

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={commit}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit()
          if (e.key === "Escape") {
            setEditValue(displayName)
            setEditing(false)
          }
        }}
        className="truncate text-sm font-semibold bg-transparent border-b border-primary/40 outline-none w-full"
      />
    )
  }

  return (
    <div className="flex items-center gap-1.5 min-w-0 group/title">
      <h3 className="truncate text-sm font-semibold">
        {displayName}
      </h3>
      <button
        onClick={(e) => {
          e.stopPropagation()
          setEditValue(displayName)
          setEditing(true)
        }}
        className="opacity-0 group-hover/title:opacity-100 transition-opacity shrink-0 p-0.5 rounded hover:bg-muted"
      >
        <Pencil className="h-3 w-3 text-muted-foreground" />
      </button>
    </div>
  )
}

function ChainCard({
  chain,
  index,
  customNames,
  onRename,
  onClick,
}: {
  chain: LineageChainSummary
  index: number
  customNames: Record<string, string>
  onRename: (id: string, name: string) => void
  onClick: () => void
}) {
  const JOB_NODE_TYPES = new Set(["sdg", "training", "upload", "eval"])
  const jobCount = chain.lineage.nodes.filter((n) => JOB_NODE_TYPES.has(n.type)).length
  const nodeCount = chain.lineage.nodes.length
  const graphHeight = nodeCount <= 2 ? 180 : nodeCount <= 4 ? 220 : 280

  return (
    <div
      className="animate-message-in rounded-xl border bg-card shadow-sm transition-all duration-200 hover:shadow-md hover:ring-2 hover:ring-primary/10 cursor-pointer"
      style={{ animationDelay: `${index * 120}ms` }}
      onClick={onClick}
    >
      <div className="flex items-center justify-between border-b px-5 py-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#daf2f2] dark:bg-[#003333]/40">
            <Workflow className="h-4 w-4 text-[#147878] dark:text-[#37a3a3]" />
          </div>
          <div className="min-w-0 flex-1">
            <EditableTitle
              chainId={chain.chain_id}
              defaultName={chain.name}
              customNames={customNames}
              onRename={onRename}
            />
            <p className="text-xs text-muted-foreground mt-0.5">
              {jobCount} {jobCount === 1 ? "job" : "jobs"}
              {" · "}
              started {formatRelativeTime(chain.created_at)}
              {" · "}
              updated {formatRelativeTime(chain.updated_at)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusBadge status={chain.latest_status} />
          <ArrowRight className="h-4 w-4 text-muted-foreground/30" />
        </div>
      </div>
      <div className="relative" style={{ height: graphHeight }}>
        <ReactFlowProvider>
          <ChainGraphPreview data={chain.lineage} />
        </ReactFlowProvider>
      </div>
    </div>
  )
}

export default function ExperimentsPage() {
  const { data: chains = [], isLoading, isError, error, refetch } = useLineageChains()
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<JobStatus[]>([])
  const [customNames, setCustomNames] = useState<Record<string, string>>(loadExperimentNames)
  const [selectedChain, setSelectedChain] = useState<LineageChainSummary | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)

  const handleRename = useCallback((chainId: string, name: string) => {
    setCustomNames((prev) => ({ ...prev, [chainId]: name }))
    saveExperimentName(chainId, name)
  }, [])

  const filteredChains = useMemo(() => {
    let result = chains
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter((c) => {
        const displayName = customNames[c.chain_id] || c.name
        return displayName.toLowerCase().includes(q)
      })
    }
    if (statusFilter.length > 0) {
      result = result.filter((c) => statusFilter.includes(c.latest_status))
    }
    return result
  }, [chains, search, statusFilter, customNames])

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  const selectedDisplayName = selectedChain
    ? (customNames[selectedChain.chain_id] || selectedChain.name)
    : ""

  return (
    <div className="space-y-6">
      <div data-tutorial="experiments-header" className="space-y-6">
        <PageHeader
          title="Experiments"
          description="End-to-end experiment tracking — from data generation to trained models"
        />

        <div className="animate-message-in rounded-xl border bg-card p-4">
          <div className="grid gap-3 md:grid-cols-3 mb-3">
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
                <Sparkles className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Data Generation</p>
                <p className="text-xs text-muted-foreground">Generates a synthetic dataset</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <GraduationCap className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Training</p>
                <p className="text-xs text-muted-foreground">Fine-tunes a model on your data</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#daf2f2] text-[#147878] dark:bg-[#003333]/40 dark:text-[#37a3a3]">
                <Workflow className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Pipeline</p>
                <p className="text-xs text-muted-foreground">Click any experiment to explore</p>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground border-t pt-3">
            Experiments are created automatically when you chain jobs together. Start from{" "}
            <Link to="/chat" className="inline-flex items-center gap-0.5 underline underline-offset-2 hover:text-foreground transition-colors">
              Chat <ArrowRight className="h-2.5 w-2.5" />
            </Link>{" "}
            or{" "}
            <Link to="/recipes" className="underline underline-offset-2 hover:text-foreground transition-colors">
              Recipes
            </Link>. Double-click any experiment name to rename it.
          </p>
        </div>
      </div>

      {isLoading ? (
        <TableSkeleton columns={3} />
      ) : chains.length > 0 ? (
        <>
          <div className="space-y-3">
            <p className="text-sm font-semibold text-muted-foreground">Filters</p>
            <div className="space-y-2.5">
              <FilterChips
                label="Status"
                options={STATUS_OPTIONS}
                selected={statusFilter}
                onChange={setStatusFilter}
              />
            </div>
            <SearchInput
              value={search}
              onChange={setSearch}
              placeholder="Search experiments..."
            />
          </div>

          {filteredChains.length > 0 ? (
            <div className="space-y-5">
              {filteredChains.map((chain, i) => (
                <ChainCard
                  key={chain.chain_id}
                  chain={chain}
                  index={i}
                  customNames={customNames}
                  onRename={handleRename}
                  onClick={() => {
                    setSelectedChain(chain)
                    setDetailOpen(true)
                  }}
                />
              ))}
            </div>
          ) : (
            <Empty className="border mt-4">
              <EmptyHeader>
                <EmptyMedia variant="icon"><Workflow /></EmptyMedia>
                <EmptyTitle>No matching experiments</EmptyTitle>
                <EmptyDescription>Try adjusting your search or filters.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </>
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Workflow /></EmptyMedia>
            <EmptyTitle>No experiments yet</EmptyTitle>
            <EmptyDescription>
              Experiments appear here when you chain jobs together — for example, data generation followed by training.
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

      <ExperimentDetailPanel
        chain={selectedChain}
        open={detailOpen}
        onOpenChange={setDetailOpen}
        displayName={selectedDisplayName}
        onRename={handleRename}
      />
    </div>
  )
}
