import { Handle, Position, type NodeProps, type Node } from "@xyflow/react"
import {
  GraduationCap,
  Sparkles,
  Upload,
  FileText,
  FlaskConical,
  Database,
  Box,
  BookOpen,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { StatusBadge } from "./status-badge"
import type { JobStatus } from "@/types/api"

export interface LineageNodeData {
  nodeType: string
  status: string
  recipe: string
  meta: Record<string, unknown>
  isTarget: boolean
  link: string
  [key: string]: unknown
}

export type LineageFlowNode = Node<LineageNodeData, "lineage">

const NODE_TYPE_CONFIG: Record<
  string,
  { label: string; icon: typeof GraduationCap; accentClass: string }
> = {
  training: {
    label: "Training",
    icon: GraduationCap,
    accentClass: "border-l-rh-blue dark:border-l-rh-blue",
  },
  sdg: {
    label: "Data Generation",
    icon: Sparkles,
    accentClass: "border-l-rh-purple dark:border-l-rh-purple",
  },
  upload: {
    label: "Upload",
    icon: Upload,
    accentClass: "border-l-rh-green dark:border-l-rh-green",
  },
  document: {
    label: "Document",
    icon: FileText,
    accentClass: "border-l-rh-yellow dark:border-l-rh-yellow",
  },
  eval: {
    label: "Eval",
    icon: FlaskConical,
    accentClass: "border-l-rh-danger dark:border-l-rh-danger",
  },
  dataset: {
    label: "Dataset",
    icon: Database,
    accentClass: "border-l-emerald-500 dark:border-l-emerald-400",
  },
  model: {
    label: "Trained Model",
    icon: Box,
    accentClass: "border-l-sky-500 dark:border-l-sky-400",
  },
  recipe: {
    label: "Recipe",
    icon: BookOpen,
    accentClass: "border-l-amber-500 dark:border-l-amber-400",
  },
}

const FALLBACK_CONFIG = {
  label: "Job",
  icon: FlaskConical,
  accentClass: "border-l-border",
}

function getMetaLines(
  nodeType: string,
  meta: Record<string, unknown>,
): string[] {
  const lines: string[] = []
  const str = (key: string) => {
    const v = meta[key]
    return typeof v === "string" && v ? v : null
  }
  const num = (key: string) => {
    const v = meta[key]
    return typeof v === "number" ? v : null
  }

  if (nodeType === "training") {
    const model = str("model")
    if (model) lines.push(model)
    const algo = str("algorithm")
    if (algo) lines.push(algo)
  } else if (nodeType === "sdg") {
    const topic = str("topic")
    if (topic) lines.push(topic)
    const n = num("num_records")
    if (n != null) lines.push(`${n.toLocaleString()} records`)
  } else if (nodeType === "upload" || nodeType === "document") {
    const filename = str("filename") || str("original_filename")
    if (filename) lines.push(filename)
  } else if (nodeType === "dataset") {
    const name = str("name")
    if (name) lines.push(name)
    const n = num("num_records")
    if (n != null) lines.push(`${n.toLocaleString()} records`)
  } else if (nodeType === "model") {
    const name = str("name")
    if (name) lines.push(name)
  } else if (nodeType === "recipe") {
    const name = str("name")
    if (name) lines.push(name)
    const jobType = str("job_type")
    if (jobType) lines.push(jobType === "sdg" ? "Data generation" : "Training")
  } else if (nodeType === "eval") {
    const model = str("model")
    if (model) lines.push(model)
  }

  return lines.slice(0, 2)
}

const JOB_TYPES = new Set(["training", "sdg", "upload", "document", "eval"])

export function LineageNode({ data }: NodeProps<LineageFlowNode>) {
  const config = NODE_TYPE_CONFIG[data.nodeType] ?? FALLBACK_CONFIG
  const Icon = config.icon
  const metaLines = getMetaLines(data.nodeType, data.meta)
  const isJob = JOB_TYPES.has(data.nodeType)

  return (
    <div
      className={cn(
        "w-[220px] rounded-lg border border-l-4 bg-card px-3 py-3 shadow-sm transition-shadow cursor-pointer",
        config.accentClass,
        data.isTarget
          ? "ring-2 ring-[var(--rh-color-accent-base-on-light,#0066cc)] shadow-md dark:ring-[var(--rh-color-accent-base-on-dark,#92c5f9)]"
          : "hover:shadow-md",
      )}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2 !w-2 !border-2 !border-card !bg-muted-foreground"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2 !w-2 !border-2 !border-card !bg-muted-foreground"
      />

      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted">
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <div className="flex flex-col gap-0.5 min-w-0">
          <span className="text-xs font-semibold leading-none">
            {config.label}
          </span>
          {isJob && data.status ? (
            <StatusBadge status={data.status as JobStatus} />
          ) : (
            <span className="text-[10px] text-muted-foreground">Artifact</span>
          )}
        </div>
      </div>

      {metaLines.length > 0 && (
        <div className="mt-2 space-y-0.5 border-t border-border/50 pt-2">
          {metaLines.map((line, i) => (
            <p
              key={i}
              className="truncate text-[11px] leading-tight text-muted-foreground"
            >
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
