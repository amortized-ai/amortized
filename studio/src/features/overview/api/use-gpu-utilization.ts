import { useQuery } from "@tanstack/react-query"
import { getGpuUtilization, ApiError } from "@/lib/api-client"
import type { GpuUtilizationResponse } from "@/types/api"

const EMPTY_RESPONSE: GpuUtilizationResponse = { nodes: [] }

export function useGpuUtilization() {
  return useQuery<GpuUtilizationResponse>({
    queryKey: ["gpu-utilization"],
    queryFn: async () => {
      try {
        return await getGpuUtilization()
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          return EMPTY_RESPONSE
        }
        if (err instanceof TypeError) {
          return EMPTY_RESPONSE
        }
        throw err
      }
    },
    refetchInterval: 10_000,
    retry: 1,
    staleTime: 5_000,
  })
}
