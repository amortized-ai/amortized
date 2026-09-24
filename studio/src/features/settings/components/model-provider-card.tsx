import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { KeyRound, Eye, EyeOff, Loader2, Trash2 } from "lucide-react"
import {
  useModelProvider,
  useAddModelProvider,
  useRemoveModelProvider,
} from "../api/use-model-provider"

const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  vertex: "Vertex (ADC)",
}

function label(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider
}

// Vertex is ADC-only: its credential is a Google application-default-credentials JSON blob
// (a multi-line file), not a single-line key string.
function isAdc(provider: string): boolean {
  return provider === "vertex"
}

function isValidCredential(provider: string, value: string): boolean {
  const v = value.trim()
  if (!v) return false
  if (isAdc(provider)) {
    try {
      const parsed = JSON.parse(v) as { type?: unknown }
      return !!parsed && typeof parsed === "object" && typeof parsed.type === "string"
    } catch {
      return false
    }
  }
  return v.length >= 8
}

const CONFIGURED_BADGE =
  "bg-[#e9f5e8] text-[#1e4f18] border-[#95d58e] dark:bg-[#0d2009]/40 dark:text-[#5ba352] dark:border-[#163b11]"

// One provider row: status badge + a credential editor (Change/Save/Cancel) mirroring the
// legacy Agent Provider section, plus a Remove action. openai/anthropic take an API key
// (masked input); vertex takes an ADC JSON blob (textarea).
function ProviderRow({ providerID, configured }: { providerID: string; configured: boolean }) {
  const add = useAddModelProvider()
  const remove = useRemoveModelProvider()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const [showKey, setShowKey] = useState(false)

  const adc = isAdc(providerID)
  const valid = isValidCredential(providerID, draft)
  const showInput = editing || !configured

  function handleSave() {
    add.mutate(
      { provider: providerID, key: draft.trim() },
      {
        onSuccess: () => {
          setDraft("")
          setShowKey(false)
          setEditing(false)
        },
      },
    )
  }

  function cancel() {
    setDraft("")
    setShowKey(false)
    setEditing(false)
  }

  const saveButton = (
    <Button
      size="sm"
      className="h-8 text-xs"
      onClick={handleSave}
      disabled={!valid || add.isPending}
      data-testid={`provider-save-${providerID}`}
    >
      {add.isPending ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : configured ? (
        "Update"
      ) : (
        "Save"
      )}
    </Button>
  )

  return (
    <div className="rounded-lg border px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <span className="text-sm font-medium">{label(providerID)}</span>
          <span className="ml-2 text-xs text-muted-foreground font-mono">{providerID}</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={configured ? CONFIGURED_BADGE : "bg-muted/50 text-muted-foreground"}
          >
            {configured ? "Configured" : "Not configured"}
          </Badge>
          {configured && !showInput && (
            <>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setEditing(true)}
              >
                Change
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 text-muted-foreground"
                aria-label={`Remove ${label(providerID)}`}
                onClick={() => remove.mutate(providerID)}
                disabled={remove.isPending}
              >
                {remove.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
              </Button>
            </>
          )}
        </div>
      </div>

      {showInput &&
        (adc ? (
          <div className="mt-2 flex flex-col gap-2">
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Paste your Vertex ADC JSON (application_default_credentials.json)"
              autoComplete="off"
              spellCheck={false}
              rows={6}
              className="flex-1 font-mono text-xs"
              data-testid={`provider-cred-${providerID}`}
            />
            <div className="flex items-center justify-end gap-2">
              {editing && (
                <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={cancel}>
                  Cancel
                </Button>
              )}
              {saveButton}
            </div>
          </div>
        ) : (
          <div className="mt-2 flex items-center gap-2">
            <Input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={configured ? "Enter a new API key…" : "Paste your API key…"}
              type={showKey ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              className="flex-1 font-mono text-xs h-8"
              data-testid={`provider-cred-${providerID}`}
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
            {editing && (
              <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={cancel}>
                Cancel
              </Button>
            )}
            {saveButton}
          </div>
        ))}

      {add.isError && (
        <p className="mt-2 text-xs text-[#c9190b]">{(add.error as Error).message}</p>
      )}
      {remove.isError && (
        <p className="mt-2 text-xs text-[#c9190b]">{(remove.error as Error).message}</p>
      )}
    </div>
  )
}

export function ModelProviderCard() {
  const { data: status, isLoading } = useModelProvider()

  // The card only applies to the hybrid gateway deployment.
  if (isLoading || !status?.available) return null

  const supported = status.supported.length > 0 ? status.supported : ["openai", "anthropic", "vertex"]
  const configured = new Set(status.configured)

  return (
    <Card id="section-model" className="scroll-mt-6">
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ffe0e0] text-[#cc0000] dark:bg-[#420303]/40 dark:text-[#e54343]">
            <KeyRound className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">Model Providers</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <p className="mb-4 text-xs text-muted-foreground">
          The credentials Morty uses to chat — and, for OpenAI/Anthropic keys, the model catalog and
          SDG teacher too. Add as many as you like; each is stored for your account only and used
          solely for your sandbox. Updating one briefly restarts Morty.
        </p>
        <div className="space-y-3">
          {supported.map((providerID) => (
            <ProviderRow key={providerID} providerID={providerID} configured={configured.has(providerID)} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
