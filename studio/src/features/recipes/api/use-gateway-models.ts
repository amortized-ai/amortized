import { useQuery } from "@tanstack/react-query"
import { listGatewayModels, type ModelsResponse } from "@/lib/api-client"

// Teacher-model catalog for recipe model pickers (gateway / direct providers).
// Same source as the SDG agent's `list_models` tool — includes gateway-served
// models like gpt-oss that the chat provider catalog doesn't carry.
export function useGatewayModels() {
  return useQuery<ModelsResponse>({
    queryKey: ["gateway-models"],
    queryFn: listGatewayModels,
    staleTime: 60_000,
  })
}
