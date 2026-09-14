import { useState, useEffect, useRef, useCallback } from "react"
import { Loader2, CircleCheck, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { getJob, getJobLogs } from "@/lib/api-client"
import type { JobStatus } from "@/types/api"

interface JobMonitorCardProps {
  jobId: string
  jobType?: string
  onDismiss?: () => void
  onComplete?: (jobId: string, jobType: string, status: string) => void
}

const TERMINAL_STATUSES: JobStatus[] = ["succeeded", "failed", "cancelled"]

function statusToProgress(status: JobStatus, elapsed: number): number {
  switch (status) {
    case "queued":
      return 10
    case "provisioning":
      return 20
    case "running":
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

function runningStageLabel(jobType: string, stageMarker: string | null): string {
  if (jobType === "EVAL") {
    switch (stageMarker) {
      case "serving":
        return "Serving the model inside the eval job (Stage 3/4)"
      case "waiting-for-endpoint":
        return "Loading model — waiting for it to be ready (Stage 3/4)"
      case "evaluating":
        return "Evaluating the model (Stage 3/4)"
      case "serve-failed":
        return "The model server failed to start — stopping"
      default:
        return "Evaluating model (Stage 3/4)"
    }
  }
  switch (jobType) {
    case "TRAINING":
      return "Training model (Stage 3/4)"
    default:
      return "Generating data (Stage 3/4)"
  }
}

function statusToStageLabel(
  status: JobStatus,
  jobType: string,
  stageMarker: string | null = null,
): string {
  switch (status) {
    case "queued":
      return "Queued (Stage 1/4)"
    case "provisioning":
      return "Provisioning resources (Stage 2/4)"
    case "running":
      return runningStageLabel(jobType, stageMarker)
    case "succeeded":
      return "Complete (Stage 4/4)"
    case "failed":
      if (stageMarker === "serve-failed")
        return "Failed — the model server did not start (check logs)"
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
  const [mlflowRunId, setMlflowRunId] = useState<string>("")
  // Eval jobs that serve the model themselves print stage markers into the
  // job logs (=== EVAL-STAGE: serving | waiting-for-endpoint | evaluating ===)
  // — surface them so the user sees where the eval is (model loading is the
  // slow part and used to be an invisible serve job).
  const isEval = jobType.toUpperCase() === "EVAL"
  const [stageMarker, setStageMarker] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const jobStartRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const completeFired = useRef(false)
  const onCompleteRef = useRef(onComplete)
  useEffect(() => { onCompleteRef.current = onComplete }, [onComplete])

  const shortId = jobId.slice(0, 8)
  const isTerminal = TERMINAL_STATUSES.includes(status)
  const progress = statusToProgress(status, elapsed)

  const pollJob = useCallback(async () => {
    try {
      const job = await getJob(jobId)
      setStatus(job.status)
      if (job.error) setError(job.error)
      if (job.mlflow_run_id) setMlflowRunId(job.mlflow_run_id)
      if (job.started_at) {
        jobStartRef.current = new Date(job.started_at).getTime()
      } else if (!jobStartRef.current && job.created_at) {
        jobStartRef.current = new Date(job.created_at).getTime()
      }
      if (TERMINAL_STATUSES.includes(job.status)) {
        if (pollRef.current) clearInterval(pollRef.current)
        if (timerRef.current) clearInterval(timerRef.current)
        if (isEval && job.status === "failed" && stageMarker === null) {
          // Best-effort: find which stage the eval died in (e.g. the
          // model server never came up) so the card can say why.
          try {
            const lines = await getJobLogs(jobId, 500)
            for (let i = lines.length - 1; i >= 0; i--) {
              const m = lines[i]?.match(/=== EVAL-STAGE: ([\w-]+) ===/)
              if (m) {
                setStageMarker(m[1] ?? null)
                break
              }
            }
          } catch {
            // log fetch is best-effort
          }
        }
        if (!completeFired.current) {
          completeFired.current = true
          onCompleteRef.current?.(jobId, jobType, job.status)
        }
        return
      }
      if (isEval && job.status === "running") {
        try {
          const lines = await getJobLogs(jobId, 5)
          for (let i = lines.length - 1; i >= 0; i--) {
            const m = lines[i]?.match(/=== EVAL-STAGE: ([\w-]+) ===/)
            if (m) {
              setStageMarker(m[1] ?? null)
              break
            }
          }
        } catch {
          // log fetch is best-effort; job status polling continues
        }
      }
    } catch {
      // Silently continue polling on transient errors
    }
  }, [jobId, jobType, isEval])

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
        {statusToStageLabel(status, jobType, stageMarker)}
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
              href={mlflowRunId ? `/models?run=${encodeURIComponent(mlflowRunId)}` : "/models"}
              className="text-primary dark:text-primary hover:underline font-medium"
            >
              View Model →
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
