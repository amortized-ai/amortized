import { useState, useEffect, useRef, useCallback } from "react"
import { Loader2, CircleCheck, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { getJob } from "@/lib/api-client"
import type { JobStatus } from "@/types/api"

interface JobMonitorCardProps {
  jobId: string
  jobType?: string
  onDismiss?: () => void
  onComplete?: (status: string) => void
}

const TERMINAL_STATUSES: JobStatus[] = ["succeeded", "failed", "cancelled"]

function statusToProgress(status: JobStatus, elapsed: number): number {
  switch (status) {
    case "queued":
      return 10
    case "provisioning":
      return 20
    case "running":
      // Ramp from 35 to 80 over ~5 minutes
      return Math.min(80, 35 + Math.floor(elapsed / 1000) * 0.15)
    case "succeeded":
      return 100
    case "failed":
    case "cancelled":
      return 100
    default:
      return 10
  }
}

function runningStageLabel(jobType: string): string {
  switch (jobType) {
    case "TRAINING":
      return "Training model (Stage 3/4)"
    case "EVAL":
      return "Evaluating model (Stage 3/4)"
    default:
      return "Generating data (Stage 3/4)"
  }
}

function statusToStageLabel(status: JobStatus, jobType: string): string {
  switch (status) {
    case "queued":
      return "Queued (Stage 1/4)"
    case "provisioning":
      return "Provisioning resources (Stage 2/4)"
    case "running":
      return runningStageLabel(jobType)
    case "succeeded":
      return "Complete (Stage 4/4)"
    case "failed":
      return "Failed"
    case "cancelled":
      return "Cancelled"
    default:
      return status
  }
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}m ${seconds}s`
}

export function JobMonitorCard({ jobId, jobType = "SDG", onDismiss, onComplete }: JobMonitorCardProps) {
  const [status, setStatus] = useState<JobStatus>("queued")
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const jobStartRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const completeFired = useRef(false)

  const shortId = jobId.slice(0, 8)
  const isTerminal = TERMINAL_STATUSES.includes(status)
  const progress = statusToProgress(status, elapsed)

  const pollJob = useCallback(async () => {
    try {
      const job = await getJob(jobId)
      setStatus(job.status)
      if (job.error) setError(job.error)
      // Always prefer server timestamp; update on every poll until we have started_at
      if (job.started_at) {
        jobStartRef.current = new Date(job.started_at).getTime()
      } else if (!jobStartRef.current && job.created_at) {
        jobStartRef.current = new Date(job.created_at).getTime()
      }
      if (TERMINAL_STATUSES.includes(job.status)) {
        if (pollRef.current) clearInterval(pollRef.current)
        if (timerRef.current) clearInterval(timerRef.current)
        if (!completeFired.current) {
          completeFired.current = true
          onComplete?.(job.status)
        }
      }
    } catch {
      // Silently continue polling on transient errors
    }
  }, [jobId, onComplete])

  // Elapsed timer — uses job's server-side start time so it survives tab switches
  useEffect(() => {
    timerRef.current = setInterval(() => {
      if (jobStartRef.current) {
        setElapsed(Date.now() - jobStartRef.current)
      }
    }, 1000)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  // Poll job status — initial poll + interval
  useEffect(() => {
    const id = setInterval(pollJob, 3000)
    pollRef.current = id
    const initial = setTimeout(() => { void pollJob() }, 0)
    return () => { clearInterval(id); clearTimeout(initial) }
  }, [pollJob])

  return (
    <div
      className={cn(
        "rounded-xl border p-4 transition-colors duration-300",
        status === "succeeded"
          ? "border-rh-green/40 dark:border-rh-green-dark bg-rh-green-light/80 dark:bg-rh-green-dark/30"
          : status === "failed" || status === "cancelled"
            ? "border-rh-danger/40 dark:border-rh-danger-dark bg-rh-danger-light/80 dark:bg-rh-danger-dark/30"
            : "border-rh-yellow dark:border-rh-yellow-dark bg-rh-yellow-light/80 dark:bg-rh-yellow-dark/30",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {status === "succeeded" ? (
            <CircleCheck className="h-4 w-4 text-rh-green dark:text-rh-green" />
          ) : status === "failed" || status === "cancelled" ? (
            <XCircle className="h-4 w-4 text-rh-danger dark:text-rh-danger" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-rh-yellow dark:text-rh-yellow" />
          )}
          <span
            className={cn(
              "text-sm font-semibold",
              status === "succeeded"
                ? "text-rh-green dark:text-rh-green"
                : status === "failed" || status === "cancelled"
                  ? "text-rh-danger dark:text-rh-danger"
                  : "text-rh-yellow dark:text-rh-yellow",
            )}
          >
            {status === "succeeded"
              ? "Job completed"
              : status === "failed"
                ? "Job failed"
                : status === "cancelled"
                  ? "Job cancelled"
                  : "Monitoring job"}
          </span>
        </div>
        {!isTerminal && onDismiss && (
          <button
            onClick={onDismiss}
            className="rounded-lg border border-rh-yellow dark:border-rh-yellow-dark bg-background dark:bg-rh-gray-80 px-3 py-1.5 text-xs font-medium text-foreground hover:bg-rh-gray-10 dark:hover:bg-rh-gray-70 transition-colors"
          >
            Skip monitoring
          </button>
        )}
      </div>

      {/* Title */}
      <p className="text-sm font-bold text-foreground mb-1">
        Monitoring {jobType} job #{shortId}
      </p>

      {/* Stage */}
      <p className="text-xs text-muted-foreground mb-2">
        {statusToStageLabel(status, jobType)}
      </p>

      {/* Progress bar */}
      <div className="h-2 w-full rounded-full bg-rh-gray-20 dark:bg-rh-gray-70 mb-2">
        <div
          className={cn(
            "h-2 rounded-full transition-all duration-500",
            status === "succeeded"
              ? "bg-rh-green"
              : status === "failed" || status === "cancelled"
                ? "bg-rh-danger"
                : "bg-gradient-to-r from-primary to-primary/80",
          )}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Stats */}
      <p className="text-xs text-muted-foreground">
        {formatElapsed(elapsed)} &bull; {Math.round(progress)}%
      </p>

      {/* Error message */}
      {error && (status === "failed" || status === "cancelled") && (
        <p className="mt-2 text-xs text-rh-danger dark:text-rh-danger">
          {error}
        </p>
      )}

      {/* Post-completion links */}
      {status === "succeeded" && (
        <div className="mt-3 flex items-center gap-3 text-xs">
          <a
            href={`/jobs?job=${encodeURIComponent(jobId)}`}
            className="text-primary dark:text-primary hover:underline font-medium"
          >
            View Job →
          </a>
          <span className="text-muted-foreground/30">|</span>
          {jobType === "TRAINING" ? (
            <a
              href="/models"
              className="text-primary dark:text-primary hover:underline font-medium"
            >
              View Model →
            </a>
          ) : jobType === "EVAL" ? (
            <a
              href={`/jobs?job=${encodeURIComponent(jobId)}`}
              className="text-primary dark:text-primary hover:underline font-medium"
            >
              View Results →
            </a>
          ) : (
            <a
              href={`/datasets?job=${encodeURIComponent(jobId)}`}
              className="text-primary dark:text-primary hover:underline font-medium"
            >
              View Dataset →
            </a>
          )}
        </div>
      )}
    </div>
  )
}
