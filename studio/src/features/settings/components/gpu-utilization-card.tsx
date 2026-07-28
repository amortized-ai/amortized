import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { CardSkeleton } from "@/components/card-skeleton"
import {
  Empty,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "@/components/ui/empty"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip"
import { Cpu, CheckCircle, AlertTriangle, AlertCircle } from "lucide-react"
import { Link } from "react-router"
import { cn } from "@/lib/utils"
import { useHealth, useGpuUtilization } from "../api/use-settings"
import type { GpuDeviceUtilization } from "@/types/api"

type GpuStatus = "healthy" | "busy" | "critical"

function getGpuStatus(device: GpuDeviceUtilization): GpuStatus {
  const memPct = (device.memory_used_mb / device.memory_total_mb) * 100
  const peak = Math.max(device.gpu_utilization_pct, memPct)
  if (peak > 90) return "critical"
  if (peak > 75) return "busy"
  return "healthy"
}

function getOverallStatus(devices: GpuDeviceUtilization[]): GpuStatus {
  let worst: GpuStatus = "healthy"
  for (const d of devices) {
    const s = getGpuStatus(d)
    if (s === "critical") return "critical"
    if (s === "busy") worst = "busy"
  }
  return worst
}

function getProgressColor(pct: number): string {
  if (pct > 90)
    return "[&>*]:bg-[#b1380b] dark:[&>*]:bg-[#f4784a]"
  if (pct > 75)
    return "[&>*]:bg-[#dca614] dark:[&>*]:bg-[#ffcc17]"
  return ""
}

function getTemperatureLabel(celsius: number): string {
  if (celsius < 50) return "cool"
  if (celsius <= 70) return "normal"
  if (celsius <= 80) return "warm"
  return "hot"
}

const statusConfig: Record<
  GpuStatus,
  { label: string; badgeClass: string; icon: React.ElementType }
> = {
  healthy: {
    label: "Healthy",
    badgeClass:
      "border-0 bg-[#e9f7df] text-[#3d7317] dark:bg-[#204d00]/20 dark:text-[#87bb62]",
    icon: CheckCircle,
  },
  busy: {
    label: "Busy",
    badgeClass:
      "border-0 bg-[#fff4cc] text-[#96640f] dark:bg-[#73480b]/20 dark:text-[#dca614]",
    icon: AlertTriangle,
  },
  critical: {
    label: "Critical",
    badgeClass:
      "border-0 bg-[#ffe3d9] text-[#b1380b] dark:bg-[#731f00]/20 dark:text-[#f4784a]",
    icon: AlertCircle,
  },
}

function StatusBadge({ status }: { status: GpuStatus }) {
  const config = statusConfig[status]
  const Icon = config.icon
  return (
    <Badge className={config.badgeClass}>
      <Icon className="h-3 w-3 mr-1" aria-hidden="true" />
      <span className="sr-only">GPU status</span>
      {config.label}
    </Badge>
  )
}

interface MetricRowProps {
  label: string
  value: number
  valueText: string
  detailText: string
  ariaLabel: string
  mounted: boolean
  barIndex: number
}

function MetricRow({
  label,
  value,
  valueText,
  detailText,
  ariaLabel,
  mounted,
  barIndex,
}: MetricRowProps) {
  return (
    <div className="group">
      <div
        className="grid grid-cols-[100px_1fr_auto] items-center gap-3"
        style={{ "--fill-delay": `${barIndex * 150}ms` } as React.CSSProperties}
      >
        <span className="text-xs text-muted-foreground w-[100px] shrink-0">
          {label}
        </span>
        <Progress
          value={mounted ? value : 0}
          className={cn(
            "[&>*]:transition-all [&>*]:duration-700 [&>*]:ease-out [&>*]:delay-[var(--fill-delay)]",
            getProgressColor(value)
          )}
          aria-label={ariaLabel}
        />
        <span className="text-xs font-mono tabular-nums text-foreground w-24 text-right">
          {valueText}
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

function DeviceCard({
  device,
  index,
  mounted,
}: {
  device: GpuDeviceUtilization
  index: number
  mounted: boolean
}) {
  const status = getGpuStatus(device)
  const memPct = (device.memory_used_mb / device.memory_total_mb) * 100
  const memGb = (device.memory_used_mb / 1024).toFixed(1)
  const totalGb = Math.round(device.memory_total_mb / 1024)

  return (
    <div
      role="group"
      aria-label={`GPU ${device.index + 1}: ${device.name}, status: ${statusConfig[status].label}`}
      className={cn(
        "animate-message-in rounded-lg border border-border/50 bg-muted/20 p-3 space-y-2.5",
        "transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md",
        status === "healthy" &&
          "hover:border-[#3d7317]/40 dark:hover:border-[#87bb62]/30",
        status === "busy" &&
          "hover:border-[#96640f]/40 dark:hover:border-[#dca614]/30",
        status === "critical" &&
          "hover:border-[#b1380b]/40 dark:hover:border-[#f4784a]/30",
        status === "critical" && "animate-status-pulse"
      )}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="flex items-center justify-between">
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="text-xs font-semibold truncate cursor-default focus-visible:ring-[3px] focus-visible:ring-ring/50 rounded-sm"
              tabIndex={0}
            >
              GPU {device.index + 1}
            </span>
          </TooltipTrigger>
          <TooltipContent>
            <span className="font-mono text-xs">{device.name}</span>
          </TooltipContent>
        </Tooltip>
        <StatusBadge status={status} />
      </div>

      <MetricRow
        label="Compute Load"
        value={device.gpu_utilization_pct}
        valueText={`${Math.round(device.gpu_utilization_pct)}%`}
        detailText={`${Math.round(device.gpu_utilization_pct)}% of processing power in use`}
        ariaLabel={`Compute load: ${Math.round(device.gpu_utilization_pct)}%`}
        mounted={mounted}
        barIndex={0}
      />
      <MetricRow
        label="Memory"
        value={memPct}
        valueText={`${Math.round(memPct)}%   ${memGb} / ${totalGb} GB`}
        detailText={`${memGb} GB / ${(device.memory_total_mb / 1024).toFixed(1)} GB VRAM`}
        ariaLabel={`Memory usage: ${Math.round(memPct)}%`}
        mounted={mounted}
        barIndex={1}
      />
      {device.temperature_celsius != null && (
        <MetricRow
          label="Temperature"
          value={Math.min(device.temperature_celsius, 100)}
          valueText={`${device.temperature_celsius}°C`}
          detailText={`${device.temperature_celsius}°C — ${getTemperatureLabel(device.temperature_celsius)}`}
          ariaLabel={`Temperature: ${device.temperature_celsius} degrees Celsius`}
          mounted={mounted}
          barIndex={2}
        />
      )}
    </div>
  )
}

function UtilizationLoadingSkeleton() {
  return (
    <div className="rounded-lg border border-border/50 bg-muted/20 p-3 space-y-2.5" aria-busy="true">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-12" />
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
      <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-2 flex-1 rounded-full" />
        <Skeleton className="h-3 w-12" />
      </div>
      <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-2 flex-1 rounded-full" />
        <Skeleton className="h-3 w-24" />
      </div>
      <div className="grid grid-cols-[100px_1fr_auto] items-center gap-3">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-2 flex-1 rounded-full" />
        <Skeleton className="h-3 w-10" />
      </div>
    </div>
  )
}

export function GpuUtilizationCard() {
  const { data: healthData, isLoading: healthLoading, isError: healthError } = useHealth({ refetchInterval: 30000 })
  const gpu = healthData?.gpu
  const gpuAvailable = gpu?.available === true && (gpu?.count ?? 0) > 0

  const {
    data: utilization,
    isLoading: utilizationLoading,
    isError: utilizationError,
  } = useGpuUtilization(gpuAvailable)

  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(id)
  }, [])

  if (healthError) return null

  if (healthLoading) return <CardSkeleton />

  if (!gpu || !gpuAvailable) {
    return (
      <Card className="animate-message-in">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <Cpu className="h-3.5 w-3.5" />
              </div>
              <CardTitle className="text-sm">Compute Accelerator</CardTitle>
            </div>
            <Badge className="bg-muted/50 text-muted-foreground border-0">
              Not detected
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <Empty className="border-0 p-6">
            <EmptyMedia variant="icon">
              <Cpu className="h-10 w-10 text-muted-foreground" />
            </EmptyMedia>
            <EmptyTitle>No accelerator detected</EmptyTitle>
            <EmptyDescription>
              {gpu?.note ||
                "Training jobs need a GPU. Connect an SSH backend with GPU access, or run on a GPU-enabled machine."}
            </EmptyDescription>
            <EmptyContent>
              <Button asChild variant="outline" size="sm">
                <Link to="/settings#section-system">Check Settings</Link>
              </Button>
            </EmptyContent>
          </Empty>
        </CardContent>
      </Card>
    )
  }

  const overallStatus =
    utilization?.devices && utilization.devices.length > 0
      ? getOverallStatus(utilization.devices)
      : "healthy"

  return (
    <Card className="animate-message-in">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
              <Cpu className="h-3.5 w-3.5" />
            </div>
            <CardTitle className="text-sm">Compute Accelerator</CardTitle>
          </div>
          {utilizationLoading ? (
            <Skeleton className="h-5 w-16 rounded-full" />
          ) : utilizationError ? (
            <Badge className="bg-muted/50 text-muted-foreground border-0">
              Metrics unavailable
            </Badge>
          ) : (
            <StatusBadge status={overallStatus} />
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {utilizationLoading ? (
          <UtilizationLoadingSkeleton />
        ) : utilizationError ? (
          <div className="rounded-lg border border-border/50 bg-muted/20 p-4 text-center space-y-2">
            <p className="text-sm text-muted-foreground">
              Live metrics are not available right now.
            </p>
            <p className="text-xs text-muted-foreground">
              GPU detection shows {gpu.count} device{(gpu.count ?? 0) > 1 ? "s" : ""} connected.
            </p>
            {gpu.devices && gpu.devices.length > 0 && (
              <div className="flex flex-wrap gap-1.5 justify-center pt-1">
                {gpu.devices.map((name, i) => (
                  <Badge key={i} variant="outline" className="font-mono text-xs">
                    {name}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        ) : (
          utilization?.devices.map((device, i) => (
            <DeviceCard
              key={device.index}
              device={device}
              index={i}
              mounted={mounted}
            />
          ))
        )}
      </CardContent>
    </Card>
  )
}
