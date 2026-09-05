import { useQuery } from "@tanstack/react-query"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-providers")

export interface ProviderModelEntry {
  providerID: string
  modelID: string
  label: string
}

export interface DynamicProvider {
  id: string
  name: string
  models: ProviderModelEntry[]
}

interface ProviderListResponse {
  all: DynamicProvider[]
  default: Record<string, string> | null
  connected: string[]
}

async function fetchProviders(): Promise<{ providers: DynamicProvider[]; connected: Set<string> }> {
  const resp = await fetch("/agent/provider", {
    headers: { "Content-Type": "application/json" },
  })
  if (!resp.ok) {
    logger.error("failed to fetch providers", { status: resp.status })
    throw new Error(`Failed to fetch providers: ${resp.status}`)
  }
  const data: ProviderListResponse = await resp.json()
  return {
    providers: data.all ?? [],
    connected: new Set(data.connected ?? []),
  }
}

export function useProviderStatus() {
  const query = useQuery({
    queryKey: ["agent", "providers"],
    queryFn: fetchProviders,
    staleTime: 60_000,
    retry: 2,
    refetchOnWindowFocus: false,
  })

  return {
    ...query,
    providers: query.data?.providers ?? [],
    connectedProviders: query.data?.connected ?? new Set<string>(),
  }
}
