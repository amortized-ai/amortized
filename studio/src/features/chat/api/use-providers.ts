import { useQuery } from "@tanstack/react-query"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-providers")

interface ProviderListResponse {
  all: Array<{ id: string }>
  default: Record<string, string> | null
  connected: string[]
}

async function fetchProviderStatus(): Promise<Set<string>> {
  const resp = await fetch("/agent/provider", {
    headers: { "Content-Type": "application/json" },
  })
  if (!resp.ok) {
    logger.error("failed to fetch providers", { status: resp.status })
    throw new Error(`Failed to fetch providers: ${resp.status}`)
  }
  const data: ProviderListResponse = await resp.json()
  return new Set(data.connected ?? [])
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
    connectedProviders: query.data ?? new Set<string>(),
  }
}
