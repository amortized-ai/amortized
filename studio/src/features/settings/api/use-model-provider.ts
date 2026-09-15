import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-model-provider")

// The per-user bring-your-own-key model provider, managed by the studio-gateway
// (hybrid deployment). `available` is true only when the gateway confirms it, and
// false only on confirmed absence (e.g. local dev), so the settings card can hide
// itself. Operational errors reject the query (react-query `isError`) so callers
// treat them as unknown rather than as confirmed absence.
export interface ModelProviderStatus {
  available: boolean
  provider: string | null
  providers: string[]
}

async function fetchModelProvider(): Promise<ModelProviderStatus> {
  const resp = await fetch("/gateway/provider", { headers: { Accept: "application/json" } })
  // Reserve `available: false` for CONFIRMED absence: a 404, or a non-JSON 200 (the
  // SPA index served for an unknown path in local dev). Any other non-OK status
  // (401/500/…) or a network failure is an operational error, not absence — let it
  // throw so react-query surfaces it as an error (an "unknown" state) instead of
  // silently activating the legacy provider flow.
  if (resp.status === 404) return { available: false, provider: null, providers: [] }
  if (!resp.ok) throw new Error(`gateway provider status ${resp.status}`)
  if (!(resp.headers.get("content-type") || "").includes("application/json")) {
    return { available: false, provider: null, providers: [] }
  }
  const data = await resp.json()
  return { available: true, provider: data.provider ?? null, providers: data.providers ?? [] }
}

export function useModelProvider() {
  return useQuery({
    queryKey: ["gateway", "model-provider"],
    queryFn: fetchModelProvider,
    staleTime: 30_000,
    retry: false,
    refetchOnWindowFocus: false,
  })
}

async function setModelProvider({ provider, key }: { provider: string; key: string }): Promise<void> {
  const resp = await fetch("/gateway/provider", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ provider, key }),
  })
  if (!resp.ok) {
    let message = `Failed to save key (${resp.status})`
    try {
      const body = await resp.json()
      if (body?.error) message = body.error
    } catch {
      /* keep the status-based message */
    }
    logger.error("set model provider failed", { provider, status: resp.status })
    throw new Error(message)
  }
}

export function useSetModelProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: setModelProvider,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["gateway", "model-provider"] })
    },
  })
}
