import { Link } from "react-router"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Cpu } from "lucide-react"
import { cn } from "@/lib/utils"
import { statusVariants } from "@/lib/status-variants"
import { useGpuAllocation } from "../api/use-gpu-allocation"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import type { GpuJobAllocation } from "@/lib/api-client"

const MAX_VISIBLE_JOBS = 5

const STATUS_CONFIG: Record<string, { label: string; className: string }> = {
  queued: { label: "Queued", className: statusVariants.neutral },
  provisioning: { label: "Provisioning", className: statusVariants.info },
  running: { label: "Running", className: `${statusVariants.info} animate-pulse` },
}

function gpuLabel(count: number): string {
  return count === 1 ? "1 GPU" : `${count} GPUs`
}

function GpuAllocationSkeleton() {
  return (
    <Card aria-busy="true" role="status">
      <CardHeader>
        <CardTitle>GPU Allocation</CardTitle>
        <Skeleton className="h-3 w-64" />
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-3">
          <Skeleton className="h-9 w-9 rounded-lg" />
          <div className="flex flex-col gap-1">
            <Skeleton className="h-8 w-12" />
            <Skeleton className="h-4 w-24" />
          </div>
        </div>
        <div className="mt-4 space-y-0 divide-y divide-border/50">
          {[0, 1].map((i) => (
            <div key={i} className="flex items-center justify-between py-2.5">
              <Skeleton className="h-4 w-40" />
              <div className="flex items-center gap-2">
                <Skeleton className="h-3 w-16" />
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-5 w-16 rounded-md" />
              </div>
            </div>
          ))}
        </div>
        <span className="sr-only">Loading GPU allocation...</span>
      </CardContent>
    </Card>
  )
}

function JobRow({ job }: { job: GpuJobAllocation }) {
  const entityName = useEntityNamesStore((s) => s.getName(job.job_id))
  const displayName = entityName || job.job_name
  const statusConfig = STATUS_CONFIG[job.status]

  return (
    <div className="flex items-center justify-between py-2.5">
      <span className="text-sm font-medium truncate min-w-0 flex-1 mr-3">
        {displayName}
      </span>
      <div className="flex items-center gap-2 shrink-0">
        {job.gpu_type && (
          <span className="text-xs text-muted-foreground">{job.gpu_type}</span>
        )}
        <span className="text-sm font-semibold tabular-nums">
          {gpuLabel(job.gpus)}
        </span>
        {statusConfig && (
          <Badge
            variant="outline"
            className={cn("font-medium", statusConfig.className)}
          >
            {statusConfig.label}
          </Badge>
        )}
      </div>
    </div>
  )
}

export function GpuAllocationCard() {
  const { data, isLoading, isError } = useGpuAllocation()

  if (isLoading) {
    return (
      <div
        className="animate-message-in"
        style={{ animationDelay: "1100ms" }}
      >
        <GpuAllocationSkeleton />
      </div>
    )
  }

  if (isError || (data && !data.available)) {
    return (
      <div
        className="animate-message-in"
        style={{ animationDelay: "1100ms" }}
      >
        <Card>
          <CardHeader>
            <CardTitle>GPU Allocation</CardTitle>
            <CardDescription>GPUs allocated to active training jobs</CardDescription>
          </CardHeader>
          <CardContent>
            <Empty className="border-0 p-6">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Cpu />
                </EmptyMedia>
                <EmptyTitle>GPU tracking not available</EmptyTitle>
                <EmptyDescription>
                  GPU allocation data requires a running backend API. Check that
                  the server is connected in Settings.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!data) return null

  const visibleJobs = data.jobs.slice(0, MAX_VISIBLE_JOBS)
  const overflowJobs = data.jobs.slice(MAX_VISIBLE_JOBS)
  const overflowGpus = overflowJobs.reduce((sum, j) => sum + j.gpus, 0)

  return (
    <div
      className="animate-message-in"
      style={{ animationDelay: "1100ms" }}
    >
      <Card>
        <CardHeader>
          <CardTitle>GPU Allocation</CardTitle>
          <CardDescription>GPUs allocated to active training jobs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
              <Cpu className="h-4 w-4" aria-hidden="true" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-3xl font-bold tabular-nums text-foreground">
                {data.total_gpus}
              </span>
              <span className="text-sm font-medium text-muted-foreground">
                GPUs in use
              </span>
            </div>
          </div>

          {data.total_gpus === 0 ? (
            <div className="mt-4 text-center space-y-1 py-4">
              <p className="text-sm font-medium text-foreground">
                No training jobs running
              </p>
              <p className="text-sm text-muted-foreground">
                GPUs are allocated when training jobs start.
              </p>
              <p className="text-sm text-muted-foreground">
                Launch a training job from a recipe or chat.
              </p>
            </div>
          ) : (
            <div className="mt-4 divide-y divide-border/50">
              {visibleJobs.map((job) => (
                <JobRow key={job.job_id} job={job} />
              ))}
              {overflowJobs.length > 0 && (
                <div className="flex items-center justify-between py-2.5">
                  <span className="text-xs text-muted-foreground">
                    +{overflowJobs.length} more {overflowJobs.length === 1 ? "job" : "jobs"} using {gpuLabel(overflowGpus)}
                  </span>
                  <Link
                    to="/jobs"
                    className="text-xs font-medium text-primary hover:underline"
                  >
                    View all jobs &rarr;
                  </Link>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
