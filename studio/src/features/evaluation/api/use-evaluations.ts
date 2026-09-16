import { useQuery } from "@tanstack/react-query"
import { getEvaluations } from "@/lib/api-client"
import type { EvaluationGroup } from "@/lib/api-client"

export function useEvaluations() {
  return useQuery<EvaluationGroup[]>({
    queryKey: ["evaluations"],
    queryFn: getEvaluations,
    refetchInterval: (query) =>
      query.state.data?.some((g) => g.evals.some((e) => e.status === "running" || e.status === "queued"))
        ? 5000
        : 30000,
  })
}
