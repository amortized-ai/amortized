import { useQuery } from "@tanstack/react-query"
import { getGpuUtilization } from "@/lib/api-client"
import type { GpuUtilizationResponse } from "@/types/api"
import { MOCK_GPU_DATA } from "./mock-gpu-data"

export function useGpuUtilization() {
  const query = useQuery<GpuUtilizationResponse>({
    queryKey: ["gpu-utilization"],
    queryFn: getGpuUtilization,
    refetchInterval: 10_000,
    retry: 1,
    staleTime: 5_000,
  })

  if (import.meta.env.DEV && query.isError) {
    return { ...query, data: MOCK_GPU_DATA, isError: false as const, isMockData: true }
  }

  return { ...query, isMockData: false }
}
