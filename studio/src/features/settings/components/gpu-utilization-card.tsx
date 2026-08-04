import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CardSkeleton } from "@/components/card-skeleton"
import { cn } from "@/lib/utils"
import { Cpu, ArrowRight } from "lucide-react"
import { Link } from "react-router"
import { useGpuUtilization } from "../api/use-settings"

function formatRelativeTime(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return "just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

const jobTypeBadgeClasses: Record<string, string> = {
  training:
    "bg-[#e0f0ff] text-[#0066cc] border-transparent dark:bg-[#003366] dark:text-[#4394e5]",
  sdg: "bg-[#ece6ff] text-[#5e40be] border-transparent dark:bg-[#1b0d33] dark:text-[#876fd4]",
}

export function GpuUtilizationCard() {
  const { data, isLoading, isError } = useGpuUtilization()

  if (isLoading) {
    return <CardSkeleton />
  }

  const unavailable = isError || !data || !data.available
  const empty = data?.available && data.jobs.length === 0

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
            <Cpu className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">GPU Utilization</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        {unavailable ? (
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">
              GPU information is not yet available.
            </p>
            <p className="text-xs text-muted-foreground">
              {isError
                ? "Check that the backend server is running."
                : "This feature requires a Kubernetes compute backend."}
            </p>
          </div>
        ) : empty ? (
          <>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-display">0</span>
                <span className="text-sm text-muted-foreground">GPUs in use</span>
              </div>
            </div>
            <div className="border-t border-border/50" />
            <div className="mt-3 space-y-1">
              <p className="text-sm text-muted-foreground">
                No jobs are currently using GPUs.
              </p>
              <p className="text-xs text-muted-foreground">
                Training and SDG jobs request GPUs when submitted.
              </p>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-display">
                  {data!.total_gpus_in_use}
                </span>
                <span className="text-sm text-muted-foreground">
                  GPU{data!.total_gpus_in_use !== 1 ? "s" : ""} in use
                </span>
              </div>
              <Button variant="link" size="sm" asChild>
                <Link to="/jobs?status=running">
                  View Jobs
                  <ArrowRight className="h-3 w-3" />
                </Link>
              </Button>
            </div>
            <div className="border-t border-border/50" />
            <div className="mt-3 space-y-0">
              {data!.jobs.map((job) => (
                <div
                  key={job.job_name}
                  className="flex items-center gap-3 py-2.5 border-b border-border/50 last:border-0 transition-all duration-200 hover:bg-muted/30"
                >
                  <Badge variant="outline" className="font-mono text-xs shrink-0">
                    {job.gpus_requested} GPU{job.gpus_requested !== 1 ? "s" : ""}
                  </Badge>
                  <div className="flex flex-col min-w-0">
                    <span className="font-medium text-sm truncate">
                      {job.job_id
                        ? job.job_name.replace(/^amortized-/, "").replace(/-[a-z0-9]{5,}$/, "")
                        : job.job_name}
                    </span>
                    <span className="text-xs text-muted-foreground font-mono truncate">
                      {job.job_name}
                    </span>
                  </div>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs shrink-0",
                      jobTypeBadgeClasses[job.job_type] ?? "",
                    )}
                  >
                    {job.job_type === "sdg"
                      ? "SDG"
                      : job.job_type === "training"
                        ? "Training"
                        : job.job_type}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs shrink-0",
                      job.status === "running"
                        ? "bg-rh-blue-light text-rh-blue-dark border-rh-blue/30 dark:bg-rh-blue-dark/30 dark:text-rh-blue dark:border-rh-blue-dark"
                        : "bg-muted text-muted-foreground border-border",
                    )}
                  >
                    {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                  </Badge>
                  {job.started_at && (
                    <span className="text-xs text-muted-foreground ml-auto shrink-0">
                      {formatRelativeTime(job.started_at)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
