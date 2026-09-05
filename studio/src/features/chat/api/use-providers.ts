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
  // Normalize at the API boundary — never trust external data shapes.
  const rawAll: unknown[] = Array.isArray(data.all) ? (data.all as unknown[]) : []
  const providers: DynamicProvider[] = rawAll
    .filter((p): p is Record<string, unknown> => !!p && typeof p === "object" && typeof (p as Record<string, unknown>).id === "string")
    .map((p) => {
      const rawModels: unknown[] = Array.isArray(p.models) ? (p.models as unknown[]) : []
      return {
        id: p.id as string,
        name: typeof p.name === "string" ? p.name : (p.id as string),
        models: rawModels
          .filter((m): m is Record<string, unknown> => !!m && typeof m === "object" && typeof (m as Record<string, unknown>).modelID === "string")
          .map((m) => ({
            providerID: typeof m.providerID === "string" ? m.providerID : (p.id as string),
            modelID: m.modelID as string,
            label: typeof m.label === "string" ? m.label : (m.modelID as string),
          })),
      }
    })
  return {
    providers,
    connected: new Set(Array.isArray(data.connected) ? data.connected : []),
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
