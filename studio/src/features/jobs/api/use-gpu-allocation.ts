import { useQuery } from "@tanstack/react-query"
import { getGpuAllocation } from "@/lib/api-client"
import type { GpuAllocationResponse } from "@/lib/api-client"

export function useGpuAllocation() {
  return useQuery<GpuAllocationResponse>({
    queryKey: ["gpu", "allocation"],
    queryFn: getGpuAllocation,
    refetchInterval: (query) => {
      const hasJobs = (query.state.data?.jobs?.length ?? 0) > 0
      return hasJobs ? 5000 : false
    },
  })
}
