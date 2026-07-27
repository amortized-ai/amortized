import { Cpu } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
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

function GpuNodeRow({ node }: { node: GpuNodeMetrics }) {
  const memoryPct = node.memory_total_mb > 0
    ? Math.round((node.memory_used_mb / node.memory_total_mb) * 100)
    : 0

  return (
    <div role="group" aria-label={`GPU ${node.index}: ${node.name}`} className="space-y-2">
      <p className="text-sm font-semibold">GPU {node.index} — {node.name}</p>

      <div className="grid grid-cols-[80px_1fr_auto] items-center gap-3">
        <span className="text-xs text-muted-foreground">Usage</span>
        <Progress
          value={node.utilization_pct}
          indicatorClassName={getThresholdColor(node.utilization_pct)}
          aria-label={`GPU ${node.index} usage: ${node.utilization_pct} percent`}
        />
        <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
          {node.utilization_pct}%
        </span>
      </div>

      <div className="grid grid-cols-[80px_1fr_auto] items-center gap-3">
        <span className="text-xs text-muted-foreground">Memory</span>
        <Progress
          value={memoryPct}
          indicatorClassName={getThresholdColor(memoryPct)}
          aria-label={`GPU ${node.index} memory: ${node.memory_used_mb} of ${node.memory_total_mb} megabytes, ${memoryPct} percent`}
        />
        <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
          {memoryPct}% ({formatGb(node.memory_used_mb)}/{formatGb(node.memory_total_mb)} GB)
        </span>
      </div>

      <div className="grid grid-cols-[80px_1fr_auto] items-center gap-3">
        <span className="text-xs text-muted-foreground">Temp</span>
        <Progress
          value={node.temperature_c}
          indicatorClassName={getThresholdColor(node.temperature_c)}
          aria-label={`GPU ${node.index} temperature: ${node.temperature_c} degrees Celsius`}
        />
        <span className="text-xs font-medium text-foreground min-w-[60px] text-right">
          {node.temperature_c} C
        </span>
      </div>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {[0, 1].map((i) => (
        <div key={i} className="space-y-2">
          <Skeleton className="h-4 w-40" />
          {[0, 1, 2].map((j) => (
            <div key={j} className="grid grid-cols-[80px_1fr_auto] items-center gap-3">
              <Skeleton className="h-3 w-12" />
              <Skeleton className="h-2 w-full rounded-full" />
              <Skeleton className="h-3 w-[60px]" />
            </div>
          ))}
          {i === 0 && <Separator className="opacity-30" />}
        </div>
      ))}
    </div>
  )
}

export function GpuUtilizationSection() {
  const { data, isLoading, isError } = useGpuUtilization()

  const nodes = data?.nodes ?? []

  return (
    <div className="px-4 lg:px-6 space-y-6">
      <div className="flex items-center gap-4">
        <div className="h-px flex-1 bg-border/50" />
        <h2 className="text-sm font-semibold text-muted-foreground/70 uppercase tracking-wider shrink-0">
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
        className="animate-message-in rounded-xl border bg-card p-5 space-y-4"
        style={{ animationDelay: "300ms" }}
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
          <div key={node.index}>
            <GpuNodeRow node={node} />
            {i < nodes.length - 1 && <Separator className="opacity-30 mt-4" />}
          </div>
        ))}
      </div>
    </div>
  )
}
