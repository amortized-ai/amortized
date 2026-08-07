import { useQuery } from "@tanstack/react-query"
import { getHealth, searchMlflowRuns } from "@/lib/api-client"
import type { HealthResponse } from "@/types/api"

export type StatusLevel = "ok" | "error" | "loading"

export interface SystemHealthStatus {
  backend: StatusLevel
  mlflow: StatusLevel
}

export function useSystemHealth(): SystemHealthStatus {
  const health = useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  const mlflow = useQuery({
    queryKey: ["health", "mlflow"],
    queryFn: () => searchMlflowRuns({ max_results: 1 }),
    refetchInterval: 30_000,
    retry: false,
  })

  const backend: StatusLevel = health.isLoading
    ? "loading"
    : health.isError
      ? "error"
      : "ok"

  const mlflowStatus: StatusLevel = mlflow.isLoading
    ? "loading"
    : mlflow.isError
      ? "error"
      : "ok"

  return { backend, mlflow: mlflowStatus }
}
