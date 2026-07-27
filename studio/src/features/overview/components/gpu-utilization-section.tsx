import { Cpu } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { useGpuUtilization } from "../api/use-gpu-utilization"
import type { GpuNodeMetrics } from "@/types/api"

function getThresholdColor(value: number): string {
  if (value > 80) return "bg-rh-danger"
  if (value >= 60) return "bg-rh-yellow"
  return "bg-rh-green"
}

function formatGb(mb: number): string {
  return (mb / 1024).toFixed(0)
}

function clampProgress(value: number): number {
  return Math.min(100, Math.max(0, value))
}

type GpuStatus = "healthy" | "busy" | "critical"

function getGpuStatus(node: GpuNodeMetrics): GpuStatus {
  const memoryPct = node.memory_total_mb > 0
    ? Math.round((node.memory_used_mb / node.memory_total_mb) * 100)
    : 0
  const maxMetric = Math.max(node.utilization_pct, memoryPct, node.temperature_c)
  if (maxMetric > 90) return "critical"
  if (maxMetric > 70) return "busy"
  return "healthy"
}

const statusConfig: Record<GpuStatus, { label: string; dotClass: string; badgeBg: string; badgeText: string }> = {
  healthy: {
    label: "Healthy",
    dotClass: "bg-rh-green",
    badgeBg: "bg-rh-green-10 dark:bg-rh-green-70/20",
    badgeText: "text-rh-green-60 dark:text-rh-green-40",
  },
  busy: {
    label: "Busy",
    dotClass: "bg-rh-yellow-30",
    badgeBg: "bg-rh-yellow-10 dark:bg-rh-yellow-70/20",
    badgeText: "text-rh-yellow-60 dark:text-rh-yellow-40",
  },
  critical: {
    label: "Critical",
    dotClass: "bg-rh-danger",
    badgeBg: "bg-rh-danger-light/30 dark:bg-rh-danger-dark/20",
    badgeText: "text-rh-danger dark:text-rh-danger-light",
  },
}

const metricTooltips = {
  computeLoad: "Percentage of GPU processing power currently in use",
  memory: "GPU video memory (VRAM) in use vs total available",
  temperature: "Current GPU core temperature in degrees Celsius",
}

function GpuNodeCard({ node, index }: { node: GpuNodeMetrics; index: number }) {
  const memoryPct = node.memory_total_mb > 0
    ? Math.round((node.memory_used_mb / node.memory_total_mb) * 100)
    : 0
  const status = getGpuStatus(node)
  const config = statusConfig[status]

  return (
    <div
      role="group"
      aria-label={`GPU ${node.index}: ${node.name}, status: ${config.label}`}
      className="animate-message-in rounded-lg border-border/50 border bg-muted/20 p-3 space-y-2 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md"
      style={{ animationDelay: `${500 + index * 100}ms` }}
    >
      <TooltipProvider>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={cn("h-2 w-2 rounded-full", config.dotClass)} aria-hidden="true" />
            <p className="text-sm font-semibold">GPU {node.index} — {node.name}</p>
          </div>
          <Badge
            variant="outline"
            className={cn("border-0 text-xs font-medium", config.badgeBg, config.badgeText)}
          >
            {config.label}
            <span className="sr-only">
              . Based on highest metric reading across compute load, memory, and temperature
            </span>
          </Badge>
        </div>

        <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs text-muted-foreground cursor-default" tabIndex={0}>
                Compute Load
              </span>
            </TooltipTrigger>
            <TooltipContent>{metricTooltips.computeLoad}</TooltipContent>
          </Tooltip>
          <Progress
            value={node.utilization_pct}
            indicatorClassName={getThresholdColor(node.utilization_pct)}
            aria-label={`GPU ${node.index} compute load: ${node.utilization_pct} percent`}
          />
          <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
            {node.utilization_pct}%
          </span>
        </div>

        <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs text-muted-foreground cursor-default" tabIndex={0}>
                Memory
              </span>
            </TooltipTrigger>
            <TooltipContent>{metricTooltips.memory}</TooltipContent>
          </Tooltip>
          <Progress
            value={memoryPct}
            indicatorClassName={getThresholdColor(memoryPct)}
            aria-label={`GPU ${node.index} memory: ${formatGb(node.memory_used_mb)} of ${formatGb(node.memory_total_mb)} gigabytes, ${memoryPct} percent`}
          />
          <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
            {memoryPct}% ({formatGb(node.memory_used_mb)}/{formatGb(node.memory_total_mb)} GB)
          </span>
        </div>

        <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs text-muted-foreground cursor-default" tabIndex={0}>
                Temperature
              </span>
            </TooltipTrigger>
            <TooltipContent>{metricTooltips.temperature}</TooltipContent>
          </Tooltip>
          <Progress
            value={clampProgress(node.temperature_c)}
            indicatorClassName={getThresholdColor(node.temperature_c)}
            aria-label={`GPU ${node.index} temperature: ${node.temperature_c} degrees Celsius`}
          />
          <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
            {node.temperature_c}°C
          </span>
        </div>
      </TooltipProvider>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {[0, 1].map((i) => (
        <div key={i} className="rounded-lg border-border/50 border bg-muted/20 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Skeleton className="h-2 w-2 rounded-full" />
              <Skeleton className="h-4 w-40" />
            </div>
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          {[0, 1, 2].map((j) => (
            <div key={j} className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-2 w-full rounded-full" />
              <Skeleton className="h-3 w-[60px]" />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export function GpuUtilizationSection() {
  const { data, isLoading, isError } = useGpuUtilization()

  const nodes = data?.nodes ?? []

  return (
    <section aria-labelledby="gpu-utilization-heading" className="px-4 lg:px-6 space-y-6">
      <div className="flex items-center gap-4">
        <div className="h-px flex-1 bg-border/50" />
        <h2 id="gpu-utilization-heading" className="text-sm font-semibold text-muted-foreground/70 uppercase tracking-wider shrink-0">
          GPU Utilization
        </h2>
        <div className="h-px flex-1 bg-border/50" />
      </div>

      {!isLoading && nodes.length > 0 && (
        <span className="sr-only">
          GPU utilization summary: {nodes.length} GPU{nodes.length !== 1 ? "s" : ""} detected.
        </span>
      )}

      <div
        className="animate-message-in rounded-xl border bg-card p-5 space-y-3"
        style={{ animationDelay: "400ms" }}
      >
        {isLoading && <LoadingSkeleton />}

        {isError && (
          <p className="text-sm text-muted-foreground text-center py-4">
            Unable to load GPU metrics.
          </p>
        )}

        {!isLoading && !isError && nodes.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-4 text-muted-foreground">
            <Cpu className="h-5 w-5" />
            <p className="text-sm">No GPU nodes detected</p>
          </div>
        )}

        {!isLoading && !isError && nodes.length > 0 && nodes.map((node, i) => (
          <GpuNodeCard key={node.index} node={node} index={i} />
        ))}
      </div>
    </section>
  )
}
