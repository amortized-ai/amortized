import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from "@/components/ui/empty"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { statusVariants } from "@/lib/status-variants"
import { Cpu, Thermometer, MemoryStick, Activity, MonitorX } from "lucide-react"
import { useGpuUtilization } from "../api/use-settings"
import type { GpuDevice } from "@/types/api"

function getProgressColor(percent: number): string {
  if (percent >= 90) return "bg-destructive"
  if (percent >= 70)
    return "bg-[#f5921b] dark:bg-[#f8ae54]"
  return "bg-primary"
}

function getTemperatureColor(celsius: number | null): string {
  if (celsius === null) return "text-muted-foreground"
  if (celsius > 90) return "text-destructive"
  if (celsius > 80) return "text-[#f5921b] dark:text-[#f8ae54]"
  return "text-muted-foreground"
}

function GpuProgressBar({
  value,
  label,
}: {
  value: number
  label: string
}) {
  return (
    <div
      className="h-1.5 w-full rounded-full bg-secondary"
      role="progressbar"
      aria-valuenow={Math.round(value)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out transition-colors duration-300 motion-reduce:transition-none",
          getProgressColor(value),
        )}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

function GpuDeviceRow({ device, index }: { device: GpuDevice; index: number }) {
  const memoryPercent =
    device.memory_total_mib > 0
      ? (device.memory_used_mib / device.memory_total_mib) * 100
      : 0

  const formatMemory = (mib: number) => {
    if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`
    return `${Math.round(mib)} MiB`
  }

  return (
    <div
      className="py-2.5 border-b border-border/50 last:border-0"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="font-mono text-xs">
          <span className="sr-only">GPU number </span>GPU {device.index}
        </Badge>
        <span className="text-sm font-medium text-foreground">{device.name}</span>
      </div>

      <div className="grid grid-cols-3 gap-3 mt-2">
        {/* Compute */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1">
            <Activity className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Compute</span>
          </div>
          <GpuProgressBar
            value={device.utilization_percent}
            label={`GPU ${device.index} compute utilization ${Math.round(device.utilization_percent)} percent`}
          />
          <span className="text-xs font-mono font-medium text-foreground">
            {Math.round(device.utilization_percent)}%
          </span>
        </div>

        {/* Memory */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1">
            <MemoryStick className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Memory</span>
          </div>
          <GpuProgressBar
            value={memoryPercent}
            label={`GPU ${device.index} memory utilization ${Math.round(memoryPercent)} percent`}
          />
          <span className="text-xs font-mono font-medium text-foreground">
            {formatMemory(device.memory_used_mib)} / {formatMemory(device.memory_total_mib)}
          </span>
        </div>

        {/* Temperature */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1">
            <Thermometer
              className={cn("h-3 w-3", getTemperatureColor(device.temperature_celsius))}
            />
            <span className="text-xs text-muted-foreground">Temp</span>
          </div>
          <span
            className={cn(
              "text-xs font-mono",
              getTemperatureColor(device.temperature_celsius),
            )}
          >
            {device.temperature_celsius !== null
              ? `${Math.round(device.temperature_celsius)}°C`
              : "—"}
          </span>
        </div>
      </div>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div aria-busy="true" role="status" className="space-y-3">
      {[0, 1].map((i) => (
        <div key={i} className="py-2.5 border-b border-border/50 last:border-0">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-12 rounded" />
            <Skeleton className="h-4 w-48 rounded" />
          </div>
          <div className="grid grid-cols-3 gap-3 mt-2">
            <div className="space-y-1.5">
              <Skeleton className="h-3 w-16 rounded" />
              <Skeleton className="h-1.5 w-full rounded-full" />
              <Skeleton className="h-3 w-8 rounded" />
            </div>
            <div className="space-y-1.5">
              <Skeleton className="h-3 w-16 rounded" />
              <Skeleton className="h-1.5 w-full rounded-full" />
              <Skeleton className="h-3 w-24 rounded" />
            </div>
            <div className="space-y-1.5">
              <Skeleton className="h-3 w-10 rounded" />
              <Skeleton className="h-3 w-10 rounded" />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export function GpuUtilizationCard() {
  const { data: gpuData, isLoading, isError, refetch } = useGpuUtilization()

  const renderContent = () => {
    if (isLoading) {
      return <LoadingSkeleton />
    }

    if (isError) {
      return (
        <Empty className="py-8">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <MonitorX className="size-6" />
            </EmptyMedia>
            <EmptyTitle>GPU Metrics Unavailable</EmptyTitle>
            <EmptyDescription>
              GPU information will appear here once the backend is connected. Check that the
              server is running.
            </EmptyDescription>
          </EmptyHeader>
          <Button variant="outline" onClick={() => void refetch()}>
            Try Again
          </Button>
        </Empty>
      )
    }

    if (!gpuData?.available) {
      return (
        <Empty className="py-8">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Cpu className="size-6" />
            </EmptyMedia>
            <EmptyTitle>No GPU Detected</EmptyTitle>
            <EmptyDescription>
              Training jobs run best with a GPU. Add a GPU node to your cluster or configure
              an SSH backend with GPU access.
            </EmptyDescription>
          </EmptyHeader>
          <Button
            variant="outline"
            onClick={() =>
              document
                .getElementById("section-system")
                ?.scrollIntoView({ behavior: "smooth", block: "start" })
            }
          >
            Check Settings
          </Button>
        </Empty>
      )
    }

    if (gpuData.available && gpuData.devices.length === 0 && gpuData.error) {
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={statusVariants.success}>
              {gpuData.device_count} GPU{gpuData.device_count > 1 ? "s" : ""} ready
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">{gpuData.error}</p>
        </div>
      )
    }

    return (
      <div className="space-y-3">
        {gpuData.devices.map((device, i) => (
          <GpuDeviceRow key={device.index} device={device} index={i} />
        ))}
      </div>
    )
  }

  const showBadge =
    gpuData?.available && gpuData.device_count > 0 && !isLoading && !isError

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
            <Cpu className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">GPU Utilization</CardTitle>
          {showBadge && (
            <Badge variant="outline" className={cn("ml-auto", statusVariants.success)}>
              {gpuData.device_count} ready
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>{renderContent()}</CardContent>
    </Card>
  )
}
