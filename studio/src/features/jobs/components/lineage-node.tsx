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
  ArrowRight,
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

interface NodeConfig {
  label: string
  icon: typeof GraduationCap
  iconBg: string
  iconColor: string
  borderClass: string
  subtitle?: string
  hoverLabel?: string
}

const NODE_TYPE_CONFIG: Record<string, NodeConfig> = {
  training: {
    label: "Training",
    icon: GraduationCap,
    iconBg: "bg-[#e0f0ff] dark:bg-[#003366]/40",
    iconColor: "text-[#0066cc] dark:text-[#4394e5]",
    borderClass: "border-[#92c5f9]/60 dark:border-[#003366]/60",
    hoverLabel: "View job",
  },
  sdg: {
    label: "Data Generation",
    icon: Sparkles,
    iconBg: "bg-[#ece6ff] dark:bg-[#1b0d33]/40",
    iconColor: "text-[#5e40be] dark:text-[#876fd4]",
    borderClass: "border-[#d0c5f4]/60 dark:border-[#21134d]/60",
    hoverLabel: "View job",
  },
  upload: {
    label: "Upload",
    icon: Upload,
    iconBg: "bg-[#e9f7df] dark:bg-[#204d00]/30",
    iconColor: "text-[#3d7317] dark:text-[#63993d]",
    borderClass: "border-[#afdc8f]/60 dark:border-[#204d00]/60",
  },
  document: {
    label: "Document",
    icon: FileText,
    iconBg: "bg-[#fff4cc] dark:bg-[#54330b]/30",
    iconColor: "text-[#96640f] dark:text-[#dca614]",
    borderClass: "border-[#ffe072]/60 dark:border-[#73480b]/60",
  },
  eval: {
    label: "Eval",
    icon: FlaskConical,
    iconBg: "bg-red-50 dark:bg-red-950/30",
    iconColor: "text-[#b1380b] dark:text-red-400",
    borderClass: "border-red-200/60 dark:border-red-900/60",
  },
  dataset: {
    label: "Dataset",
    icon: Database,
    iconBg: "bg-[#ece6ff] dark:bg-[#1b0d33]/40",
    iconColor: "text-[#5e40be] dark:text-[#876fd4]",
    borderClass: "border-[#d0c5f4]/60 dark:border-[#21134d]/60",
    subtitle: "View dataset",
  },
  model: {
    label: "Trained Model",
    icon: Box,
    iconBg: "bg-[#e0f0ff] dark:bg-[#003366]/40",
    iconColor: "text-[#0066cc] dark:text-[#4394e5]",
    borderClass: "border-[#92c5f9]/60 dark:border-[#003366]/60",
    subtitle: "View model",
  },
  recipe: {
    label: "Recipe",
    icon: BookOpen,
    iconBg: "bg-amber-50 dark:bg-amber-950/30",
    iconColor: "text-amber-600 dark:text-amber-500",
    borderClass: "border-amber-200/60 dark:border-amber-800/60",
  },
}

const FALLBACK_CONFIG: NodeConfig = {
  label: "Job",
  icon: FlaskConical,
  iconBg: "bg-muted",
  iconColor: "text-muted-foreground",
  borderClass: "border-border",
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
  const isRunning = data.status === "running" || data.status === "provisioning"

  return (
    <div
      className={cn(
        "group/node w-[220px] rounded-lg border bg-card px-4 py-3.5 shadow-sm transition-all duration-200 cursor-pointer",
        config.borderClass,
        "hover:ring-2 hover:ring-primary/20 hover:shadow-md",
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
        <div className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-md transition-colors",
          config.iconBg,
          isRunning && "animate-pulse",
        )}>
          <Icon className={cn("h-3.5 w-3.5", config.iconColor)} />
        </div>
        <div className="flex flex-col gap-1 min-w-0 flex-1">
          <span className="text-xs font-semibold leading-none">
            {config.label}
          </span>
          {isJob && data.status ? (
            <div className="flex flex-col gap-0.5">
              <StatusBadge status={data.status as JobStatus} />
              <span className="inline-flex items-center gap-0.5 text-[10px] text-primary">
                View job <ArrowRight className="h-2.5 w-2.5" />
              </span>
            </div>
          ) : config.subtitle ? (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-primary">
              {config.subtitle} <ArrowRight className="h-2.5 w-2.5" />
            </span>
          ) : (
            <span className="text-[10px] text-muted-foreground">Artifact</span>
          )}
        </div>
        <ArrowRight className="h-3 w-3 text-muted-foreground/30 shrink-0 group-hover/node:text-muted-foreground/60 transition-colors" />
      </div>

      {metaLines.length > 0 && (
        <div className="mt-2.5 space-y-0.5 border-t border-border/50 pt-2">
          {metaLines.map((line, i) => (
            <p
              key={i}
              className="truncate text-xs leading-tight text-muted-foreground"
            >
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
