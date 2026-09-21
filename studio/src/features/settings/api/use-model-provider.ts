import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { getBaseUrl } from "@/lib/api-client"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-model-provider")

// The per-user bring-your-own-keys model providers, managed by the studio-gateway (hybrid
// deployment). A user can configure several providers at once; `configured` lists the ones
// with a credential stored. `available` is true only when the gateway confirms it, and
// false only on confirmed absence (e.g. local dev), so the settings card can hide itself.
// Operational errors reject the query (react-query `isError`) so callers treat them as
// unknown rather than as confirmed absence.
export interface ModelProviderStatus {
  available: boolean
  supported: string[]
  configured: string[]
}

async function fetchModelProvider(): Promise<ModelProviderStatus> {
  const resp = await fetch(`${getBaseUrl()}/gateway/provider`, { headers: { Accept: "application/json" } })
  // Reserve `available: false` for CONFIRMED absence: a 404, or a non-JSON 200 (the SPA
  // index served for an unknown path in local dev). Any other non-OK status (401/500/…) or
  // a network failure is an operational error, not absence — let it throw so react-query
  // surfaces it as an error (an "unknown" state) instead of silently hiding the card.
  if (resp.status === 404) return { available: false, supported: [], configured: [] }
  if (!resp.ok) throw new Error(`gateway provider status ${resp.status}`)
  if (!(resp.headers.get("content-type") || "").includes("application/json")) {
    return { available: false, supported: [], configured: [] }
  }
  const data = await resp.json()
  return { available: true, supported: data.supported ?? [], configured: data.configured ?? [] }
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

function invalidateProviderQueries(queryClient: ReturnType<typeof useQueryClient>) {
  // The configured set (settings) and the chat model picker (`/agent/provider`, driven by
  // opencode's actual sandbox providers) both change when a provider is added/removed.
  void queryClient.invalidateQueries({ queryKey: ["gateway", "model-provider"] })
  void queryClient.invalidateQueries({ queryKey: ["agent", "providers"] })
}

async function readError(resp: Response, fallback: string): Promise<string> {
  try {
    const body = await resp.json()
    if (body?.error) return body.error as string
  } catch {
    /* keep the status-based message */
  }
  return `${fallback} (${resp.status})`
}

async function addModelProvider({ provider, key }: { provider: string; key: string }): Promise<void> {
  const resp = await fetch(`${getBaseUrl()}/gateway/provider`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ provider, key }),
  })
  if (!resp.ok) {
    const message = await readError(resp, "Failed to save credential")
    logger.error("add model provider failed", { provider, status: resp.status })
    throw new Error(message)
  }
}

// Add or update ONE provider's credential (non-destructive: other providers are preserved).
export function useAddModelProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: addModelProvider,
    onSuccess: () => invalidateProviderQueries(queryClient),
  })
}

async function removeModelProvider(provider: string): Promise<void> {
  const resp = await fetch(`${getBaseUrl()}/gateway/provider/${encodeURIComponent(provider)}`, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  })
  if (!resp.ok) {
    const message = await readError(resp, "Failed to remove provider")
    logger.error("remove model provider failed", { provider, status: resp.status })
    throw new Error(message)
  }
}

export function useRemoveModelProvider() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: removeModelProvider,
    onSuccess: () => invalidateProviderQueries(queryClient),
  })
}
