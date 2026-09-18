import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { KeyRound, Eye, EyeOff, Loader2, CheckCircle2 } from "lucide-react"
import { useModelProvider, useSetModelProvider } from "../api/use-model-provider"

const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
}

function label(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider
}

export function ModelProviderCard() {
  const { data: status, isLoading } = useModelProvider()
  const save = useSetModelProvider()
  const [selection, setSelection] = useState<string | null>(null)
  const [key, setKey] = useState("")
  const [showKey, setShowKey] = useState(false)
  const [savedProvider, setSavedProvider] = useState<string | null>(null)

  // The card only applies to the hybrid gateway deployment.
  if (isLoading || !status?.available) return null

  const providers = status.providers.length > 0 ? status.providers : ["openai", "anthropic"]
  const current = status.provider
  const selected = selection ?? current ?? providers[0] ?? "openai"

  function handleSave() {
    save.mutate(
      { provider: selected, key: key.trim() },
      {
        onSuccess: () => {
          setSavedProvider(selected)
          setKey("")
          setShowKey(false)
        },
      },
    )
  }

  return (
    <Card id="section-model" className="scroll-mt-6">
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ffe0e0] text-[#cc0000] dark:bg-[#420303]/40 dark:text-[#e54343]">
            <KeyRound className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">Model Provider</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            The API key Morty uses to chat. It is stored for your account only and used solely for your
            sandbox. Updating it briefly restarts Morty.
          </p>
          {current ? (
            <Badge
              variant="outline"
              className="shrink-0 bg-[#e9f5e8] text-[#1e4f18] border-[#95d58e] dark:bg-[#0d2009]/40 dark:text-[#5ba352] dark:border-[#163b11]"
            >
              {label(current)} · key set
            </Badge>
          ) : (
            <Badge variant="outline" className="shrink-0 bg-muted/50 text-muted-foreground">
              No key set
            </Badge>
          )}
        </div>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Select value={selected} onValueChange={setSelection}>
            <SelectTrigger className="h-8 w-full sm:w-44" data-testid="model-provider-select">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {providers.map((p) => (
                <SelectItem key={p} value={p}>
                  {label(p)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="flex flex-1 items-center gap-2">
            <Input
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder={current ? "Enter a new API key…" : "Paste your API key…"}
              type={showKey ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              className="flex-1 font-mono text-xs h-8"
              data-testid="model-provider-key"
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
            <Button
              size="sm"
              className="h-8 text-xs"
              onClick={handleSave}
              disabled={key.trim().length < 8 || save.isPending}
              data-testid="model-provider-save"
            >
              {save.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : current ? "Update" : "Save"}
            </Button>
          </div>
        </div>

        {save.isError && (
          <p className="mt-2 text-xs text-[#c9190b]">{(save.error as Error).message}</p>
        )}
        {save.isSuccess && savedProvider && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-[#1e4f18] dark:text-[#5ba352]">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Saved. Morty is restarting with your {label(savedProvider)} key (about a minute).
          </p>
        )}
      </CardContent>
    </Card>
  )
}
