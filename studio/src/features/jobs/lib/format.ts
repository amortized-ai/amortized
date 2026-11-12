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

// LightEval reports each metric twice: "<task>.<m>" and "all.<m>", the average over the
// run's tasks. A job runs one benchmark, so the two are equal; the average is hidden
// unless it is the score itself (a suite).
export function isRunAverage(name: string, primary: string | null | undefined): boolean {
  return name.startsWith("all.") && name !== primary
}

// A benchmark metric's value: 4 significant digits, no unit (scores, perplexities, stderr).
export function formatBenchmarkValue(value: number): string {
  return String(Number(value.toPrecision(4)))
}
