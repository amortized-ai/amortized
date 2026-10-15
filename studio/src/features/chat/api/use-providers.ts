import { useQuery } from "@tanstack/react-query"
import { getBaseUrl } from "@/lib/api-client"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-providers")

export interface ProviderModel {
  id: string
  name: string
}

export interface ProviderEntry {
  id: string
  name?: string
  models?: ProviderModel[]
}

interface ProviderListResponse {
  all: ProviderEntry[]
  default: Record<string, string> | null
  connected: string[]
}

interface ProviderStatus {
  connected: Set<string>
  catalog: ProviderEntry[]
}

async function fetchProviderStatus(): Promise<ProviderStatus> {
  const resp = await fetch(`${getBaseUrl()}/agent/provider`, {
    headers: { "Content-Type": "application/json" },
  })
  if (!resp.ok) {
    logger.error("failed to fetch providers", { status: resp.status })
    throw new Error(`Failed to fetch providers: ${resp.status}`)
  }
  const data: ProviderListResponse = await resp.json()
  return {
    connected: new Set(data.connected ?? []),
    catalog: Array.isArray(data.all) ? data.all : [],
  }
}

export function useProviderStatus() {
  const query = useQuery({
    queryKey: ["agent", "providers"],
    queryFn: fetchProviderStatus,
    staleTime: 60_000,
    retry: 2,
    refetchOnWindowFocus: false,
  })

  return {
    ...query,
    connectedProviders: query.data?.connected ?? new Set<string>(),
    providerCatalog: query.data?.catalog ?? [],
  }
}
