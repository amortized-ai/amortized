import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(
  date: string,
  options?: { includeTime?: boolean },
): string {
  const d = new Date(date)
  const fmt = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    ...(options?.includeTime && {
      hour: "numeric",
      minute: "2-digit",
    }),
  })
  return fmt.format(d)
}

export function formatRelativeTime(date: string | null): string {
  if (!date) return "Never"
  const now = Date.now()
  const then = new Date(date).getTime()
  const diffMs = now - then
  if (diffMs < 0) return "Just now"
  const diffMin = Math.floor(diffMs / 60_000)
  if (diffMin < 1) return "Just now"
  if (diffMin < 60) return `${diffMin} min ago`
  const diffHrs = Math.floor(diffMin / 60)
  if (diffHrs < 24) return `${diffHrs} ${diffHrs === 1 ? "hour" : "hours"} ago`
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(new Date(date))
}
