import { useQuery } from "@tanstack/react-query"
import { getGpuAllocation } from "@/lib/api-client"
import type { GpuAllocationData } from "@/lib/api-client"

export function useGpuAllocation() {
  return useQuery<GpuAllocationData>({
    queryKey: ["jobs", "gpu-allocation"],
    queryFn: () => getGpuAllocation(),
    refetchInterval: (query) => {
      const hasActive = (query.state.data?.jobs.length ?? 0) > 0
      return hasActive ? 5000 : 30_000
    },
  })
}
