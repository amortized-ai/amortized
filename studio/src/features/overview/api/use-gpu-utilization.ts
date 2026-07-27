import { useQuery } from "@tanstack/react-query"
import { getGpuUtilization } from "@/lib/api-client"
import type { GpuUtilizationResponse } from "@/types/api"

export function useGpuUtilization() {
  return useQuery<GpuUtilizationResponse>({
    queryKey: ["gpu-utilization"],
    queryFn: getGpuUtilization,
    refetchInterval: 10_000,
    retry: 1,
    staleTime: 5_000,
  })
}
