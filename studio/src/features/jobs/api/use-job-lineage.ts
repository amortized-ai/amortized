import { useQuery } from "@tanstack/react-query"
import { getJobLineage } from "@/lib/api-client"
import type { LineageResponse } from "@/types/api"

export function useJobLineage(jobId: string | null) {
  return useQuery<LineageResponse>({
    queryKey: ["jobs", jobId, "lineage"],
    queryFn: () => getJobLineage(jobId!),
    enabled: !!jobId,
  })
}
