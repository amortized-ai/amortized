import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Wifi,
  Database,
} from "lucide-react"
import { getHealth, searchMlflowRuns } from "@/lib/api-client"
import { statusVariants, statusIconVariants } from "@/lib/status-variants"
import type { HealthResponse, MlflowRunsSearchResponse } from "@/types/api"

type CheckStatus = "ok" | "error" | "loading"

interface CheckItem {
  label: string
  status: CheckStatus
  hint: string
  icon: React.ReactNode
}

function StatusIcon({ status }: { status: CheckStatus }) {
  switch (status) {
    case "ok":
      return <CheckCircle2 className={`h-5 w-5 ${statusIconVariants.ok}`} data-testid="status-ok" />
    case "error":
      return <XCircle className={`h-5 w-5 ${statusIconVariants.error}`} data-testid="status-error" />
    case "loading":
      return <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" data-testid="status-loading" />
  }
}

function statusBadgeVariant(status: CheckStatus) {
  switch (status) {
    case "ok":
      return statusVariants.success
    case "error":
      return statusVariants.error
    case "loading":
      return ""
  }
}

function statusLabel(status: CheckStatus) {
  switch (status) {
    case "ok":
      return "Ready"
    case "error":
      return "Unavailable"
    case "loading":
      return "Checking…"
  }
}

export function PrerequisitesCard() {
  const health = useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: false,
  })

  const mlflow = useQuery<MlflowRunsSearchResponse>({
    queryKey: ["mlflow", "health-check"],
    queryFn: () => searchMlflowRuns({ max_results: 1 }),
    retry: false,
  })

  const backendStatus: CheckStatus = health.isLoading
    ? "loading"
    : health.isError
      ? "error"
      : "ok"

  const mlflowStatus: CheckStatus = mlflow.isLoading
    ? "loading"
    : mlflow.isError
      ? "error"
      : "ok"

  const checks: CheckItem[] = [
    {
      label: "Backend Connection",
      status: backendStatus,
      hint:
        backendStatus === "error"
          ? "Cannot reach the backend API. Is the server running?"
          : "Connected to the backend API.",
      icon: <Wifi className="h-4 w-4" />,
    },
    {
      label: "MLflow Connection",
      status: mlflowStatus,
      hint:
        mlflowStatus === "error"
          ? "Cannot reach MLflow. Check the MLflow tracking server."
          : "Connected to MLflow.",
      icon: <Database className="h-4 w-4" />,
    },
  ]

  const isLoading = checks.some((c) => c.status === "loading")
  const allReady = !isLoading && checks.every((c) => c.status === "ok")
  const errorCount = checks.filter((c) => c.status === "error").length

  return (
    <Card data-testid="prerequisites-card">
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
            <Wifi className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">Service Connections</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {checks.map((check) => (
            <div
              key={check.label}
              className="flex items-center gap-3 min-w-0"
              data-testid={`prereq-${check.label.toLowerCase().replace(/ /g, "-")}`}
            >
              <StatusIcon status={check.status} />
              <span className="flex items-center gap-2 text-sm text-muted-foreground">
                {check.icon}
              </span>
              <span className="text-sm font-medium">{check.label}</span>
              <Badge
                variant="outline"
                className={statusBadgeVariant(check.status)}
                data-testid={`prereq-badge-${check.label.toLowerCase().replace(/ /g, "-")}`}
              >
                {statusLabel(check.status)}
              </Badge>
              <span className="ml-auto text-xs text-muted-foreground truncate max-w-[50%]" title={check.hint}>
                {check.hint}
              </span>
            </div>
          ))}
        </div>

        {/* Status summary */}
        {isLoading && (
          <div className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-2 mt-3">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              Checking connections...
            </span>
          </div>
        )}
        {!isLoading && allReady && (
          <div className="flex items-center gap-2 rounded-lg bg-[#e9f7df] px-3 py-2 mt-3 dark:bg-[#204d00]/40">
            <CheckCircle2 className="h-4 w-4 text-[#3d7317] dark:text-[#87bb62]" />
            <span className="text-sm font-medium text-[#3d7317] dark:text-[#87bb62]">
              All services connected
            </span>
          </div>
        )}
        {!isLoading && errorCount > 0 && (
          <div className="flex items-center gap-2 rounded-lg bg-[#ffe3d9] px-3 py-2 mt-3 dark:bg-[#731f00]/40">
            <XCircle className="h-4 w-4 text-[#b1380b] dark:text-[#f0561d]" />
            <span className="text-sm font-medium text-[#b1380b] dark:text-[#f0561d]">
              {errorCount} of {checks.length} services unavailable
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
