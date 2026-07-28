import { useQuery } from "@tanstack/react-query"
import { getGpuUtilization } from "@/lib/api-client"
import type { GpuUtilizationResponse } from "@/types/api"

const EMPTY_RESPONSE: GpuUtilizationResponse = { nodes: [] }

export function useGpuUtilization() {
  return useQuery<GpuUtilizationResponse>({
    queryKey: ["gpu-utilization"],
    queryFn: async () => {
      try {
        return await getGpuUtilization()
      } catch {
        return EMPTY_RESPONSE
      }
    },
    refetchInterval: 10_000,
    retry: 0,
    staleTime: 5_000,
  })
}
