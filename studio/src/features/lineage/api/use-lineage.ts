import { useQuery } from "@tanstack/react-query"
import { getLineageChains } from "@/lib/api-client"
import type { LineageChainSummary } from "@/types/api"

export function useLineageChains() {
  return useQuery<LineageChainSummary[]>({
    queryKey: ["lineage", "chains"],
    queryFn: getLineageChains,
  })
}
