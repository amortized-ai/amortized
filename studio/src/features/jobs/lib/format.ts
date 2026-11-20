import type { JobType } from "@/types/api"

export const JOB_TYPE_LABELS: Record<JobType, string> = {
  training: "Training",
  sdg: "SDG",
  upload: "Upload",
  eval: "Eval",
  serve: "Serve",
}

// Display label for a job type, e.g. "eval" -> "Eval"; "Job" for an unknown or missing type.
export function jobTypeLabel(type: string | undefined): string {
  return JOB_TYPE_LABELS[(type ?? "").toLowerCase() as JobType] ?? "Job"
}

export function formatDuration(
  startedAt: string | null,
  completedAt: string | null,
): string {
  if (!startedAt) return "—"

  const start = new Date(startedAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  const seconds = Math.floor((end - start) / 1000)

  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  return `${hours}h ${remainingMinutes}m`
}

