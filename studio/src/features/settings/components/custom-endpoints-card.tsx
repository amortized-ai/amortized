import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { DeleteEntityDialog } from "@/components/delete-entity-dialog"
import { Eye, EyeOff, Loader2, Plug, RotateCw, Trash2 } from "lucide-react"
import {
  useAddCustomEndpoint,
  useCustomEndpoints,
  useRemoveCustomEndpoint,
} from "../api/use-custom-endpoints"

export const CUSTOM_ENDPOINTS_SECTION_ID = "section-endpoints"

// Same rules the gateway enforces (studio-gateway validateEndpoint); checked here so the form
// says what is wrong before the request.
const NAME_RE = /^[a-z](?:[a-z0-9-]{0,30}[a-z0-9])?$/
const RESERVED_NAMES = new Set([
  "nvidia",
  "openai",
  "openrouter",
  "anthropic",
  "maas",
  "enmaas",
  "enmaas-anthropic",
  "enmaas-openai",
  "vertex",
  "gateway",
])

function endpointNameError(name: string): string | null {
  if (!NAME_RE.test(name)) {
    return "Use lowercase letters, digits and hyphens, starting with a letter (max 32)."
  }
  if (RESERVED_NAMES.has(name)) return "That is the name of a built-in provider."
  return null
}

function endpointUrlError(value: string): string | null {
  let url: URL
  try {
    url = new URL(value.trim())
  } catch {
    return "Enter a full URL, e.g. https://…/v1"
  }
  const host = url.hostname.toLowerCase()
  const inCluster = host.endsWith(".svc") || host.endsWith(".svc.cluster.local")
  if (!(url.protocol === "https:" || (url.protocol === "http:" && inCluster))) {
    return "Use HTTPS (plain HTTP only for in-cluster *.svc hosts): the key is sent to this URL."
  }
  if (url.username || url.password || url.search || url.hash) {
    return "Remove credentials or query parameters from the URL; put the key in the API key field."
  }
  return null
}

function EndpointForm({ initialName, onDone }: { initialName?: string; onDone?: () => void }) {
  const add = useAddCustomEndpoint()
  const [name, setName] = useState(initialName ?? "")
  const [baseURL, setBaseURL] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [showKey, setShowKey] = useState(false)

  const nameError = name ? endpointNameError(name.trim()) : null
  const urlError = baseURL ? endpointUrlError(baseURL) : null
  const keyError = apiKey && /\s/.test(apiKey.trim()) ? "The key must not contain spaces." : null
  const valid = !!name && !!baseURL && !!apiKey.trim() && !nameError && !urlError && !keyError

  function handleSave() {
    add.mutate(
      { name: name.trim(), baseURL: baseURL.trim().replace(/\/+$/, ""), apiKey: apiKey.trim() },
      {
        onSuccess: () => {
          setName(initialName ?? "")
          setBaseURL("")
          setApiKey("")
          setShowKey(false)
          onDone?.()
        },
      },
    )
  }

  const fieldError = nameError ?? urlError ?? keyError ?? (add.isError ? (add.error as Error).message : null)

  return (
    <div className="flex flex-col gap-2">
      <Input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Name, e.g. my-vllm"
        disabled={!!initialName}
        autoComplete="off"
        spellCheck={false}
        className="font-mono text-xs h-8"
        data-testid="endpoint-name"
      />
      <Input
        value={baseURL}
        onChange={(e) => setBaseURL(e.target.value)}
        placeholder="Base URL, e.g. https://…/v1"
        autoComplete="off"
        spellCheck={false}
        className="font-mono text-xs h-8"
        data-testid="endpoint-baseurl"
      />
      <div className="flex items-center gap-2">
        <Input
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={initialName ? "Enter the API key again…" : "API key"}
          type={showKey ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          className="flex-1 font-mono text-xs h-8"
          data-testid="endpoint-key"
        />
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0"
          onClick={() => setShowKey(!showKey)}
          aria-label={showKey ? "Hide key" : "Show key"}
        >
          {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
        </Button>
        {onDone && (
          <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={onDone}>
            Cancel
          </Button>
        )}
        <Button
          size="sm"
          className="h-8 text-xs"
          onClick={handleSave}
          disabled={!valid || add.isPending}
          data-testid="endpoint-save"
        >
          {add.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : initialName ? "Update" : "Add"}
        </Button>
      </div>
      {fieldError && <p className="text-xs text-[#c9190b]">{fieldError}</p>}
    </div>
  )
}

function EndpointRow({ name, baseURL }: { name: string; baseURL: string }) {
  const remove = useRemoveCustomEndpoint()
  const [editing, setEditing] = useState(false)
  const [confirmingRemove, setConfirmingRemove] = useState(false)

  return (
    <div className="rounded-lg border px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className="text-sm font-medium font-mono">{name}</span>
          <span className="ml-2 text-xs text-muted-foreground font-mono break-all">{baseURL}</span>
        </div>
        {!editing && (
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => setEditing(true)}>
              Change
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-muted-foreground"
              aria-label={`Remove ${name}`}
              onClick={() => setConfirmingRemove(true)}
              disabled={remove.isPending}
            >
              {remove.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
            </Button>
          </div>
        )}
      </div>
      {editing && (
        <div className="mt-2">
          <EndpointForm initialName={name} onDone={() => setEditing(false)} />
        </div>
      )}
      {remove.isError && <p className="mt-2 text-xs text-[#c9190b]">{(remove.error as Error).message}</p>}
      <DeleteEntityDialog
        open={confirmingRemove}
        entityType="custom endpoint"
        entityName={name}
        description={`This removes "${name}" and its stored API key. Jobs that use it fail until you add it again, and your backend restarts briefly.`}
        isPending={remove.isPending}
        onCancel={() => setConfirmingRemove(false)}
        onConfirm={() => remove.mutate(name, { onSettled: () => setConfirmingRemove(false) })}
      />
    </div>
  )
}

function CardHeading() {
  return (
    <CardHeader>
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ffe0e0] text-[#cc0000] dark:bg-[#420303]/40 dark:text-[#e54343]">
          <Plug className="h-3.5 w-3.5" />
        </div>
        <CardTitle className="text-sm">Custom Endpoints</CardTitle>
      </div>
    </CardHeader>
  )
}

export function CustomEndpointsCard() {
  const { data: status, isLoading, isError, error, refetch, isFetching } = useCustomEndpoints()
  const available = !isLoading && !!status?.available

  // The chat's API-key warning links here (settings#section-endpoints).
  useEffect(() => {
    if (available && window.location.hash === `#${CUSTOM_ENDPOINTS_SECTION_ID}`) {
      document.getElementById(CUSTOM_ENDPOINTS_SECTION_ID)?.scrollIntoView({ block: "start" })
    }
  }, [available])

  // A failed request is not the same as "not available here": show it, with a retry.
  if (isError) {
    return (
      <Card id={CUSTOM_ENDPOINTS_SECTION_ID} className="scroll-mt-6">
        <CardHeading />
        <CardContent className="flex items-center justify-between gap-3">
          <p className="text-xs text-[#c9190b]">Could not load custom endpoints: {(error as Error).message}</p>
          <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => void refetch()} disabled={isFetching}>
            {isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />}
            Retry
          </Button>
        </CardContent>
      </Card>
    )
  }

  // Only the hybrid gateway deployment stores custom endpoints.
  if (!available || !status) return null

  return (
    <Card id={CUSTOM_ENDPOINTS_SECTION_ID} className="scroll-mt-6">
      <CardHeading />
      <CardContent>
        <p className="mb-4 text-xs text-muted-foreground">
          OpenAI-compatible endpoints that need an API key, such as a model you serve yourself. Eval and
          SDG jobs reach each one by its name, so tell Morty the name, never the key. Keys are stored for
          your account only and are not shown again. Saving or removing one briefly restarts your backend.
        </p>
        <div className="space-y-3">
          {status.endpoints.map((endpoint) => (
            <EndpointRow key={endpoint.name} name={endpoint.name} baseURL={endpoint.baseURL} />
          ))}
          <div className="rounded-lg border border-dashed px-4 py-3">
            <EndpointForm />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
