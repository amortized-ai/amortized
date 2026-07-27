import { useState, useEffect } from "react"
import { Cpu, CheckCircle, AlertTriangle, AlertCircle, type LucideIcon } from "lucide-react"
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

function formatGbDecimal(mb: number): string {
  return (mb / 1024).toFixed(1)
}

function clampProgress(value: number): number {
  return Math.min(100, Math.max(0, value))
}

function getTemperatureAdjective(temp: number): string {
  if (temp > 80) return "hot"
  if (temp >= 70) return "warm"
  if (temp >= 50) return "normal"
  return "cool"
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

const statusConfig: Record<GpuStatus, {
  label: string
  badgeBg: string
  badgeText: string
  icon: LucideIcon
}> = {
  healthy: {
    label: "Healthy",
    badgeBg: "bg-rh-green-10 dark:bg-rh-green-70/20",
    badgeText: "text-rh-green-60 dark:text-rh-green-40",
    icon: CheckCircle,
  },
  busy: {
    label: "Busy",
    badgeBg: "bg-rh-yellow-10 dark:bg-rh-yellow-70/20",
    badgeText: "text-rh-yellow-60 dark:text-rh-yellow-40",
    icon: AlertTriangle,
  },
  critical: {
    label: "Critical",
    badgeBg: "bg-rh-danger-light/30 dark:bg-rh-danger-dark/20",
    badgeText: "text-rh-danger dark:text-rh-danger-light",
    icon: AlertCircle,
  },
}

const hoverBorderClass: Record<GpuStatus, string> = {
  healthy: "hover:border-rh-green/40 dark:hover:border-rh-green-40/30",
  busy: "hover:border-rh-yellow/40 dark:hover:border-rh-yellow-40/30",
  critical: "hover:border-rh-danger/40 dark:hover:border-rh-danger-light/30",
}

const metricTooltips = {
  computeLoad: "Percentage of GPU processing power currently in use",
  memory: "GPU video memory (VRAM) in use vs total available",
  temperature: "Current GPU core temperature in degrees Celsius",
}

function MetricRow({
  label,
  tooltipText,
  value,
  displayValue,
  detailText,
  barIndex,
  mounted,
  ariaLabel,
}: {
  label: string
  tooltipText: string
  value: number
  displayValue: string
  detailText: string
  barIndex: number
  mounted: boolean
  ariaLabel: string
}) {
  return (
    <div className="group">
      <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-xs text-muted-foreground cursor-default" tabIndex={0}>
              {label}
            </span>
          </TooltipTrigger>
          <TooltipContent>{tooltipText}</TooltipContent>
        </Tooltip>
        <div style={{ "--fill-delay": `${barIndex * 150}ms` } as React.CSSProperties}>
          <Progress
            value={mounted ? value : 0}
            indicatorClassName={cn(
              getThresholdColor(value),
              "duration-700 ease-out [transition-delay:var(--fill-delay)]"
            )}
            aria-label={ariaLabel}
          />
        </div>
        <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
          {displayValue}
        </span>
      </div>
      <div className="grid grid-cols-[100px_1fr] gap-3">
        <span />
        <span className="text-[11px] text-muted-foreground/0 group-hover:text-muted-foreground/60 transition-all duration-200 h-0 group-hover:h-4 overflow-hidden">
          {detailText}
        </span>
      </div>
    </div>
  )
}

function GpuNodeCard({ node, index }: { node: GpuNodeMetrics; index: number }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(id)
  }, [])

  const memoryPct = node.memory_total_mb > 0
    ? Math.round((node.memory_used_mb / node.memory_total_mb) * 100)
    : 0
  const status = getGpuStatus(node)
  const config = statusConfig[status]

  return (
    <div
      role="group"
      aria-label={`GPU ${node.index + 1}: ${node.name}, status: ${config.label}`}
      className={cn(
        "animate-message-in rounded-lg border-border/50 border bg-muted/20 p-3 space-y-2",
        "transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md",
        hoverBorderClass[status]
      )}
      style={{ animationDelay: `${500 + index * 100}ms` }}
    >
      <TooltipProvider>
        <div className="flex items-center justify-between">
          <Tooltip>
            <TooltipTrigger asChild>
              <p className="text-sm font-semibold cursor-default" tabIndex={0}>
                GPU {node.index + 1}
              </p>
            </TooltipTrigger>
            <TooltipContent>
              <span>{node.name}</span>
            </TooltipContent>
          </Tooltip>
          <Badge
            variant="outline"
            className={cn(
              "border-0 text-xs font-medium gap-1",
              config.badgeBg,
              config.badgeText,
              status === "critical" && "animate-status-pulse"
            )}
          >
            <config.icon className="h-3 w-3" aria-hidden="true" />
            {config.label}
            <span className="sr-only">
              . Based on highest metric reading across compute load, memory, and temperature
            </span>
          </Badge>
        </div>

        <MetricRow
          label="Compute Load"
          tooltipText={metricTooltips.computeLoad}
          value={node.utilization_pct}
          displayValue={`${node.utilization_pct}%`}
          detailText={`${node.utilization_pct}% of processing power in use`}
          barIndex={0}
          mounted={mounted}

          ariaLabel={`GPU ${node.index + 1} compute load: ${node.utilization_pct} percent`}
        />

        <MetricRow
          label="Memory"
          tooltipText={metricTooltips.memory}
          value={memoryPct}
          displayValue={`${memoryPct}% (${formatGb(node.memory_used_mb)}/${formatGb(node.memory_total_mb)} GB)`}
          detailText={`${formatGbDecimal(node.memory_used_mb)} GB / ${formatGbDecimal(node.memory_total_mb)} GB VRAM`}
          barIndex={1}
          mounted={mounted}

          ariaLabel={`GPU ${node.index + 1} memory: ${formatGb(node.memory_used_mb)} of ${formatGb(node.memory_total_mb)} gigabytes, ${memoryPct} percent`}
        />

        <MetricRow
          label="Temperature"
          tooltipText={metricTooltips.temperature}
          value={clampProgress(node.temperature_c)}
          displayValue={`${node.temperature_c}°C`}
          detailText={`${node.temperature_c}°C — ${getTemperatureAdjective(node.temperature_c)}`}
          barIndex={2}
          mounted={mounted}

          ariaLabel={`GPU ${node.index + 1} temperature: ${node.temperature_c} degrees Celsius`}
        />
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
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-5 w-20 rounded-full" />
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
  const { data, isLoading, isError, isMockData } = useGpuUtilization()

  const nodes = data?.nodes ?? []

  return (
    <section aria-labelledby="gpu-utilization-heading" className="px-4 lg:px-6 space-y-6">
      <div className="flex items-center gap-4">
        <div className="h-px flex-1 bg-border/50" />
        <h2 id="gpu-utilization-heading" className="text-sm font-semibold text-muted-foreground/70 uppercase tracking-wider shrink-0">
          GPU Utilization
        </h2>
        {isMockData && (
          <span className="text-xs text-muted-foreground/50 font-normal normal-case tracking-normal">
            Demo data
          </span>
        )}
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
