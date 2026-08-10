import { useMemo, useCallback } from "react"
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useReactFlow,
  ReactFlowProvider,
  type Edge as FlowEdge,
} from "@xyflow/react"
import { Graph, layout } from "@dagrejs/dagre"
import "@xyflow/react/dist/style.css"
import { useJobLineage } from "../api/use-job-lineage"
import { LineageNode, type LineageFlowNode } from "./lineage-node"
import type { LineageResponse } from "@/types/api"

const NODE_WIDTH = 220
const NODE_HEIGHT = 130

const nodeTypes = { lineage: LineageNode } as const

function buildLayoutedGraph(data: LineageResponse) {
  const g = new Graph({ directed: true })
  g.setGraph({ rankdir: "LR", nodesep: 60, ranksep: 100 })
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

function LineageFlow({ data }: { data: LineageResponse }) {
  const { fitView } = useReactFlow()

  const { nodes, edges } = useMemo(() => buildLayoutedGraph(data), [data])

  const onInit = useCallback(() => {
    setTimeout(() => fitView({ padding: 0.3, duration: 300 }), 50)
  }, [fitView])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onInit={onInit}
      fitView
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
      minZoom={0.5}
      maxZoom={1.5}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} className="!bg-background" />
      <Controls
        showInteractive={false}
        className="!border-border !bg-card !shadow-sm [&>button]:!border-border [&>button]:!bg-card [&>button]:!fill-foreground"
      />
      <MiniMap
        className="!border-border !bg-card !shadow-sm"
        maskColor="rgba(0,0,0,0.1)"
        nodeColor="var(--color-muted)"
      />
    </ReactFlow>
  )
}

export function LineageTab({ jobId }: { jobId: string }) {
  const { data, isLoading, error } = useJobLineage(jobId)

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        <div className="flex flex-col items-center gap-3">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
          <span>Loading lineage...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Failed to load lineage data.
      </div>
    )
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        No lineage data — this job has no parent or child jobs.
      </div>
    )
  }

  return (
    <div className="h-full w-full">
      <ReactFlowProvider>
        <LineageFlow data={data} />
      </ReactFlowProvider>
    </div>
  )
}
