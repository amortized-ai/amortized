import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { getBaseUrl } from "@/lib/api-client"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-custom-endpoints")

// The per-user custom endpoints (OpenAI-compatible URL + API key) that eval and SDG jobs reach
// by name, managed by the studio-gateway. The gateway returns names and base URLs only, never
// the keys. `available` is false only on confirmed absence (404, or the SPA index served in
// local dev), so the card can hide itself; operational errors reject the query instead.
export interface CustomEndpoint {
  name: string
  baseURL: string
}

export interface CustomEndpointsStatus {
  available: boolean
  endpoints: CustomEndpoint[]
}

const QUERY_KEY = ["gateway", "custom-endpoints"]

async function fetchCustomEndpoints(): Promise<CustomEndpointsStatus> {
  const resp = await fetch(`${getBaseUrl()}/gateway/endpoints`, { headers: { Accept: "application/json" } })
  if (resp.status === 404) return { available: false, endpoints: [] }
  if (!resp.ok) throw new Error(`gateway endpoints status ${resp.status}`)
  if (!(resp.headers.get("content-type") || "").includes("application/json")) {
    return { available: false, endpoints: [] }
  }
  const data = await resp.json()
  return { available: true, endpoints: data.endpoints ?? [] }
}

export function useCustomEndpoints() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: fetchCustomEndpoints,
    staleTime: 30_000,
    retry: false,
    refetchOnWindowFocus: false,
  })
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

export interface NewCustomEndpoint {
  name: string
  baseURL: string
  apiKey: string
}

async function addCustomEndpoint(endpoint: NewCustomEndpoint): Promise<void> {
  const resp = await fetch(`${getBaseUrl()}/gateway/endpoints`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(endpoint),
  })
  if (!resp.ok) {
    const message = await readError(resp, "Failed to save endpoint")
    logger.error("add custom endpoint failed", { name: endpoint.name, status: resp.status })
    throw new Error(message)
  }
}

// Add or replace ONE endpoint (the others are preserved).
export function useAddCustomEndpoint() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: addCustomEndpoint,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

async function removeCustomEndpoint(name: string): Promise<void> {
  const resp = await fetch(`${getBaseUrl()}/gateway/endpoints/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  })
  if (!resp.ok) {
    const message = await readError(resp, "Failed to remove endpoint")
    logger.error("remove custom endpoint failed", { name, status: resp.status })
    throw new Error(message)
  }
}

export function useRemoveCustomEndpoint() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: removeCustomEndpoint,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}
