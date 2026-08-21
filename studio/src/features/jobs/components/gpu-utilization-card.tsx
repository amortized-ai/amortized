import { Link } from "react-router"
import { Cpu, AlertTriangle } from "lucide-react"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { StatusBadge } from "@/features/jobs/components/status-badge"
import { useGpuAllocation } from "@/features/jobs/api/use-gpu-allocation"
import { cn } from "@/lib/utils"
import type { JobStatus } from "@/types/api"

const MAX_VISIBLE_JOBS = 5

function GpuUtilizationCardLoading() {
  return (
    <Card className="shadow-xs" aria-busy="true" role="status">
      <CardHeader>
        <CardTitle>
          <Skeleton className="h-5 w-40" />
        </CardTitle>
        <CardDescription>
          <Skeleton className="h-4 w-72" />
        </CardDescription>
      </CardHeader>
      <CardContent>
        <span className="sr-only">Loading GPU allocation data...</span>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="space-y-2 rounded-lg border border-border bg-card p-3">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-6 w-20" />
            </div>
          ))}
        </div>
        <div className="mt-4">
          <Skeleton className="h-2.5 w-full rounded-full" />
        </div>
        <div className="mt-3 space-y-2">
          <Skeleton className="h-5 w-full rounded" />
          <Skeleton className="h-5 w-full rounded" />
        </div>
      </CardContent>
    </Card>
  )
}

function GpuUtilizationCardNoGpu({ refetch }: { refetch?: () => void; reason?: string }) {
  return (
    <Card className="shadow-xs">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Cpu className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
          GPU Allocation
        </CardTitle>
        <CardDescription>Resource allocation across active training jobs</CardDescription>
      </CardHeader>
      <CardContent>
        {refetch ? (
          <Empty className="border-0 p-6">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <AlertTriangle className="h-10 w-10 text-destructive" />
              </EmptyMedia>
              <EmptyTitle>Cannot reach the backend server</EmptyTitle>
              <EmptyDescription>
                GPU allocation data is unavailable. Is the backend API running?
              </EmptyDescription>
            </EmptyHeader>
            <Button variant="outline" onClick={refetch}>
              Try again
            </Button>
          </Empty>
        ) : (
          <Empty className="border-0 p-6">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Cpu className="h-10 w-10" />
              </EmptyMedia>
              <EmptyTitle>No GPU detected on this server</EmptyTitle>
              <EmptyDescription>
                Training jobs need a GPU. Connect an SSH backend with GPU access or deploy on a
                GPU-enabled node.
              </EmptyDescription>
            </EmptyHeader>
            <Button variant="outline" asChild>
              <Link to="/settings#infrastructure">Check Settings</Link>
            </Button>
          </Empty>
        )}
      </CardContent>
    </Card>
  )
}

export function GpuUtilizationCard() {
  const { data, isLoading, isError, refetch } = useGpuAllocation()

  if (isLoading) return <GpuUtilizationCardLoading />

  if (isError) {
    return <GpuUtilizationCardNoGpu refetch={() => void refetch()} />
  }

  if (!data?.available) {
    return <GpuUtilizationCardNoGpu />
  }

  const { total_gpus, allocated_gpus, total_memory_requested_gib, jobs } = data
  const hasActiveJobs = jobs.length > 0
  const allocationPercent = total_gpus > 0 ? Math.round((allocated_gpus / total_gpus) * 100) : 0
  const isFullyAllocated = total_gpus > 0 && allocated_gpus >= total_gpus

  const visibleJobs = jobs.slice(0, MAX_VISIBLE_JOBS)
  const remainingCount = jobs.length - MAX_VISIBLE_JOBS

  const percentLabel = isFullyAllocated
    ? "All GPUs allocated"
    : allocated_gpus === 0
      ? "All GPUs available"
      : `${allocationPercent}% allocated`

  return (
    <Card className="shadow-xs transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Cpu className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
          GPU Allocation
        </CardTitle>
        <CardDescription>Resource allocation across active training jobs</CardDescription>
        {hasActiveJobs && (
          <CardAction>
            <div className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full bg-primary animate-[gpu-pulse_2s_ease-in-out_infinite]"
              />
              <span className="text-xs text-muted-foreground">live</span>
              <span className="sr-only">Polling for updates</span>
            </div>
          </CardAction>
        )}
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-border bg-card p-3 space-y-1">
            <p className="text-xs text-muted-foreground">GPUs</p>
            <p className="text-lg font-bold tabular-nums text-foreground">
              {allocated_gpus} / {total_gpus}
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 space-y-1">
            <p className="text-xs text-muted-foreground">Memory</p>
            <p className="text-lg font-bold tabular-nums text-foreground">
              {total_memory_requested_gib > 0 ? `${Math.round(total_memory_requested_gib)} GiB` : "0 GiB"}
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 space-y-1">
            <p className="text-xs text-muted-foreground">Active Jobs</p>
            <p className="text-lg font-bold tabular-nums text-foreground">{jobs.length}</p>
          </div>
        </div>

        <div className="mt-4">
          <div
            className="h-2.5 w-full rounded-full bg-secondary"
            role="meter"
            aria-label="GPU allocation"
            aria-valuemin={0}
            aria-valuemax={total_gpus}
            aria-valuenow={allocated_gpus}
            aria-valuetext={`${allocated_gpus} of ${total_gpus} GPUs allocated`}
          >
            <div
              className={cn(
                "h-full rounded-full bg-primary transition-all duration-300",
                allocationPercent === 0 && "w-0",
              )}
              style={{ width: allocationPercent > 0 ? `${allocationPercent}%` : undefined }}
            />
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{percentLabel}</p>
        </div>

        <Separator className="my-3" />

        {hasActiveJobs ? (
          <div>
            <ul className="space-y-0">
              {visibleJobs.map((job) => (
                <li
                  key={job.job_id}
                  className="flex items-center gap-2 border-b border-border/50 py-2.5 last:border-0"
                >
                  <StatusBadge status={job.status as JobStatus} />
                  <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                    {job.job_name}
                  </span>
                  <span className="shrink-0 text-sm font-medium font-mono tabular-nums text-muted-foreground">
                    {job.gpus_requested}&times; GPU
                  </span>
                </li>
              ))}
            </ul>
            {remainingCount > 0 && (
              <Link
                to="/jobs?status=running"
                className="mt-2 inline-block text-xs text-primary hover:underline"
              >
                +{remainingCount} more in Jobs
              </Link>
            )}
          </div>
        ) : (
          <Empty className="border-0 p-6">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Cpu className="h-10 w-10" />
              </EmptyMedia>
              <EmptyTitle>All GPUs are available</EmptyTitle>
              <EmptyDescription>
                GPUs are allocated when training jobs run. Start a training job from Recipes or
                Chat.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
      </CardContent>
    </Card>
  )
}
