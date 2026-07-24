import { useMutation, useQueryClient } from "@tanstack/react-query"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-provider-auth")

interface AuthorizeInput {
  providerID: string
  apiKey: string
}

async function authorizeProvider({ providerID, apiKey }: AuthorizeInput): Promise<void> {
  const resp = await fetch(`/agent/provider/${encodeURIComponent(providerID)}/oauth/authorize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method: 0, inputs: { key: apiKey } }),
  })
  if (!resp.ok) {
    const text = await resp.text()
    logger.error("provider authorize failed", { providerID, status: resp.status, body: text })
    throw new Error(`Failed to authorize ${providerID}: ${resp.status}`)
  }
}

export function useProviderAuthorize() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: authorizeProvider,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["agent", "providers"] })
    },
  })
}
