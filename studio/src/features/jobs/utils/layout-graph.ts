import { Graph, layout } from "@dagrejs/dagre"
import type { Edge as FlowEdge } from "@xyflow/react"
import { LineageNode, type LineageFlowNode } from "../components/lineage-node"
import type { LineageResponse } from "@/types/api"

export const NODE_WIDTH = 220
export const NODE_HEIGHT = 130

export const nodeTypes = { lineage: LineageNode } as const

interface LayoutOptions {
  nodesep?: number
  ranksep?: number
}

export function buildLayoutedGraph(
  data: LineageResponse,
  opts: LayoutOptions = {},
) {
  const { nodesep = 50, ranksep = 90 } = opts
  const g = new Graph({ directed: true })
  g.setGraph({ rankdir: "LR", nodesep, ranksep })
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
        link: node.link,
      },
    }
  })

  const runningStatuses = new Set(["running", "provisioning"])
  const edges: FlowEdge[] = data.edges.map((edge) => {
    const sourceNode = data.nodes.find((n) => n.id === edge.source)
    const isRunning = sourceNode ? runningStatuses.has(sourceNode.status) : false
    return {
      id: `${edge.source}->${edge.target}`,
      source: edge.source,
      target: edge.target,
      animated: isRunning,
      style: { stroke: "var(--color-border)", strokeWidth: 2 },
    }
  })

  return { nodes, edges }
}
