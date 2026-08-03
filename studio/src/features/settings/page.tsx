import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Plus,
  Server,
  Zap,
  Cpu,
  Eye,
  EyeOff,
  ArrowRight,
  Bot,
  ExternalLink,
} from "lucide-react"
import { Link } from "react-router"
import { PageHeader } from "@/components/page-header"
import { SearchInput } from "@/components/search-input"
import { TableSkeleton } from "@/components/table-skeleton"
import { PrerequisitesCard } from "./components/prerequisites-card"
import {
  useHealth,
  useConfig,
  useGatewayRoutes,
} from "./api/use-settings"
import { useSettingsStore } from "@/stores/settings-store"
import { PROVIDER_CATALOG } from "@/features/chat/models"
import { useProviderStatus } from "@/features/chat/api/use-providers"
import { useProviderAuthorize } from "@/features/chat/api/use-provider-auth"

function ConfigRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2.5 border-b border-border/50 last:border-0">
      <span className="text-sm text-muted-foreground shrink-0">{label}</span>
      <span className={`text-sm text-right min-w-0 break-all ${mono ? "font-mono text-xs" : ""}`}>
        {value || <span className="text-muted-foreground/50">Not configured</span>}
      </span>
    </div>
  )
}


function ProviderKeyInput({ providerID }: { providerID: string }) {
  const { connectedProviders } = useProviderStatus()
  const authorize = useProviderAuthorize()
  const [draft, setDraft] = useState("")
  const [showKey, setShowKey] = useState(false)
  const isConnected = connectedProviders.has(providerID)

  function handleSave() {
    authorize.mutate(
      { providerID, apiKey: draft },
      { onSuccess: () => { setDraft(""); setShowKey(false) } },
    )
  }

  if (isConnected && !draft) {
    return (
      <div className="flex items-center gap-2 mt-2">
        <code className="flex-1 rounded-md bg-muted/50 px-3 py-1.5 text-xs font-mono text-muted-foreground">
          {showKey ? "••••••••••••••••" : "API key configured"}
        </code>
        <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => setDraft(" ")}>
          Change
        </Button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 mt-2">
      <Input
        value={draft.trim()}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="Enter API key..."
        type={showKey ? "text" : "password"}
        className="flex-1 font-mono text-xs h-8"
      />
      <Button
        variant="ghost"
        size="sm"
        className="h-8 w-8 p-0"
        onClick={() => setShowKey(!showKey)}
      >
        {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
      </Button>
      <Button
        size="sm"
        className="h-8 text-xs"
        onClick={handleSave}
        disabled={!draft.trim() || authorize.isPending}
      >
        {authorize.isPending ? "Saving..." : "Save"}
      </Button>
      {draft && (
        <Button
          variant="ghost"
          size="sm"
          className="h-8 text-xs"
          onClick={() => setDraft("")}
        >
          Cancel
        </Button>
      )}
    </div>
  )
}

function AgentProviderSection() {
  const { enabledProviders, toggleProvider } = useSettingsStore()
  const { connectedProviders, isLoading } = useProviderStatus()

  return (
    <Card id="section-agent" className="scroll-mt-6">
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ffe0e0] text-[#cc0000] dark:bg-[#420303]/40 dark:text-[#e54343]">
            <Bot className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">Agent Provider</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground mb-4">
          Enable LLM providers for the chat agent. Models from enabled providers appear in the chat model selector.
        </p>
        {isLoading ? (
          <TableSkeleton columns={2} rows={3} />
        ) : (
          <div className="space-y-3">
            {Object.entries(PROVIDER_CATALOG).map(([providerID, info]) => {
              const isEnabled = enabledProviders.includes(providerID)
              const isConnected = connectedProviders.has(providerID)

              return (
                <div
                  key={providerID}
                  className="rounded-lg border px-4 py-3 transition-all duration-200 hover:bg-muted/30"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Switch
                        checked={isEnabled}
                        onCheckedChange={() => toggleProvider(providerID)}
                        size="sm"
                      />
                      <div>
                        <span className="text-sm font-medium">{info.label}</span>
                        <span className="ml-2 text-xs text-muted-foreground font-mono">{providerID}</span>
                      </div>
                    </div>
                    <Badge
                      variant="outline"
                      className={
                        isConnected
                          ? "bg-[#e9f5e8] text-[#1e4f18] border-[#95d58e] dark:bg-[#0d2009]/40 dark:text-[#5ba352] dark:border-[#163b11]"
                          : "bg-muted/50 text-muted-foreground"
                      }
                    >
                      {isConnected ? "Connected" : "Not connected"}
                    </Badge>
                  </div>
                  {info.requiresApiKey && isEnabled && (
                    <ProviderKeyInput providerID={providerID} />
                  )}
                  {!info.requiresApiKey && (
                    <p className="text-xs text-muted-foreground mt-1.5 ml-9">
                      Pre-configured via platform credentials
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function SettingsPage() {
  const { data: config, isLoading: configLoading } = useConfig()
  const { data: healthData } = useHealth({ refetchInterval: 30000 })
  const { data: routes = [], isLoading: routesLoading } = useGatewayRoutes()
  const [routeSearch, setRouteSearch] = useState("")

  const filteredRoutes = useMemo(() => {
    if (!routeSearch.trim()) return routes
    const q = routeSearch.toLowerCase()
    return routes.filter((r) =>
      r.name.toLowerCase().includes(q) ||
      r.model.provider.toLowerCase().includes(q) ||
      r.model.name.toLowerCase().includes(q)
    )
  }, [routes, routeSearch])

  const gpu = healthData?.gpu

  return (
    <div className="space-y-6 overflow-x-hidden max-w-full">
      <div data-tutorial="settings-nav" className="space-y-6">
        <PageHeader title="Settings" description="Manage connections, compute backends, and API keys" />

        {/* Jump-to nav */}
        <div className="animate-message-in grid gap-2.5 md:grid-cols-3">
        <button
          type="button"
          onClick={() => document.getElementById("section-system")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          className="group flex items-center gap-3 rounded-xl border bg-card px-4 py-3.5 text-left transition-all duration-200 hover:border-[#92c5f9] hover:shadow-sm cursor-pointer dark:hover:border-[#003366]"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#e0f0ff] text-[#0066cc] transition-colors duration-200 group-hover:bg-[#b9dafc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
            <Server className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">System</p>
            <p className="text-xs text-muted-foreground">Backends, MLflow, GPU</p>
          </div>
          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5" />
        </button>

        <button
          type="button"
          onClick={() => document.getElementById("section-agent")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          className="group flex items-center gap-3 rounded-xl border bg-card px-4 py-3.5 text-left transition-all duration-200 hover:border-[#ffb3b3] hover:shadow-sm cursor-pointer dark:hover:border-[#730303]"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#ffe0e0] text-[#cc0000] transition-colors duration-200 group-hover:bg-[#ffb3b3] dark:bg-[#420303]/40 dark:text-[#e54343]">
            <Bot className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">Agent</p>
            <p className="text-xs text-muted-foreground">LLM providers for chat</p>
          </div>
          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5" />
        </button>

        <button
          type="button"
          onClick={() => document.getElementById("section-gateway")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          className="group flex items-center gap-3 rounded-xl border bg-card px-4 py-3.5 text-left transition-all duration-200 hover:border-[#d0c5f4] hover:shadow-sm cursor-pointer dark:hover:border-[#21134d]"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#ece6ff] text-[#5e40be] transition-colors duration-200 group-hover:bg-[#d0c5f4] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
            <Zap className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">AI Gateway</p>
            <p className="text-xs text-muted-foreground">LLM routes for SDG</p>
          </div>
          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5" />
        </button>

      </div>

      {/* Setup guide banner */}
      <div className="animate-message-in flex items-center justify-between rounded-xl border bg-card px-4 py-3" style={{ animationDelay: "100ms" }}>
        <p className="text-xs text-muted-foreground">
          Values are set via <code className="text-[11px] font-mono bg-muted px-1 py-0.5 rounded">AMORTIZED_*</code> env vars.{" "}
          <Link to="/chat" className="inline-flex items-center gap-0.5 underline underline-offset-2 hover:text-foreground transition-colors">
            Ask Morty <ArrowRight className="h-2.5 w-2.5" />
          </Link>
        </p>
        <a
          href="https://github.com/amortized-ai/amortized#readme"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all duration-200 hover:bg-muted hover:border-border"
        >
          Full setup guide <ArrowRight className="h-3 w-3" />
        </a>
      </div>
      </div>

      {/* Connection Status */}
      <div id="section-system" className="scroll-mt-6">
        <PrerequisitesCard />
      </div>

      {/* GPU Status */}
      {gpu && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <Cpu className="h-3.5 w-3.5" />
              </div>
              <CardTitle className="text-sm">GPU</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            {gpu.available && gpu.count && gpu.count > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-[#e9f7df] text-[#204d00] border-[#afdc8f] dark:bg-[#204d00]/40 dark:text-[#63993d] dark:border-[#204d00]">
                    {gpu.count} GPU{gpu.count > 1 ? "s" : ""} available
                  </Badge>
                </div>
                {gpu.devices && gpu.devices.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {gpu.devices.map((device, i) => (
                      <Badge key={i} variant="outline" className="font-mono text-xs">
                        {device}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">No GPU detected</p>
                <p className="text-xs text-muted-foreground">
                  {gpu.note || "Training jobs require a GPU. Configure an SSH backend with GPU access or run on a GPU-enabled machine."}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Platform Configuration */}
      {configLoading ? (
        <TableSkeleton columns={2} rows={6} />
      ) : config && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <Server className="h-3.5 w-3.5" />
              </div>
              <CardTitle className="text-sm">Platform Configuration</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div>
              <ConfigRow label="MLflow Tracking URI" value={config.mlflow_tracking_uri} mono />
              {config.mlflow_gateway_uri && (
                <ConfigRow label="MLflow Gateway URI" value={config.mlflow_gateway_uri} mono />
              )}
              <ConfigRow label="Default Compute Backend" value={config.default_compute_backend} />
              <div className="flex items-start justify-between gap-4 py-2.5 border-b border-border/50">
                <span className="text-sm text-muted-foreground shrink-0">Available Backends</span>
                <div className="flex flex-wrap gap-1.5 justify-end">
                  {config.available_backends.length > 0 ? (
                    config.available_backends.map((b) => (
                      <Badge key={b} variant="outline" className="text-xs">{b}</Badge>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground/50">None</span>
                  )}
                </div>
              </div>
              <ConfigRow label="Compute Namespace" value={config.compute_namespace} />
              <ConfigRow label="Image Registry" value={config.image_registry} mono />
              <ConfigRow label="Version" value={config.version} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Agent Provider */}
      <AgentProviderSection />

      {/* AI Gateway */}
      <Card id="section-gateway" className="scroll-mt-6">
        <CardHeader>
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
              <Zap className="h-3.5 w-3.5" />
            </div>
            <CardTitle className="text-sm">AI Gateway</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {config?.mlflow_gateway_uri && (
            <div className="rounded-lg border border-[#95d58e] bg-[#e9f5e8] px-4 py-3 dark:border-[#163b11] dark:bg-[#0d2009]/40">
              <p className="text-xs text-[#1e4f18] dark:text-[#5ba352]">
                SDG and evaluation jobs auto-route through the gateway. Manage endpoints in the{" "}
                <a
                  href={config.mlflow_gateway_uri}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline underline-offset-2 hover:text-[#163b11] dark:hover:text-[#7ec975]"
                >
                  MLflow AI Gateway
                </a>.
              </p>
            </div>
          )}

          {/* Endpoints */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="text-sm font-medium">Endpoints</h4>
                <p className="text-xs text-muted-foreground mt-0.5">
                  LLM endpoints for SDG and evaluation jobs.
                </p>
              </div>
              <a
                href={config?.mlflow_gateway_uri || "/mlflow/"}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all duration-200 hover:bg-muted hover:border-border"
                data-testid="add-gateway-route"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Endpoint
                <ExternalLink className="h-3 w-3 text-muted-foreground" />
              </a>
            </div>
            {routesLoading ? (
              <TableSkeleton columns={3} rows={3} />
            ) : (
              <>
                {routes.length > 3 && (
                  <div className="mb-3">
                    <SearchInput
                      value={routeSearch}
                      onChange={setRouteSearch}
                      placeholder="Search endpoints..."
                    />
                  </div>
                )}
                {filteredRoutes.length > 0 ? (
                  <div className="space-y-2">
                    {filteredRoutes.map((route) => (
                      <div
                        key={route.name}
                        className="flex items-center gap-3 rounded-lg border px-3 py-2.5 transition-all duration-200 hover:bg-muted/30 hover:border-border"
                      >
                        <span className="font-medium text-sm">{route.name}</span>
                        <Badge variant="outline" className="text-xs">{route.route_type}</Badge>
                        <span className="text-xs text-muted-foreground font-mono">
                          {route.model.provider}/{route.model.name}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : routes.length > 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No endpoints match &ldquo;{routeSearch}&rdquo;
                  </p>
                ) : (
                  <div className="text-center py-4 space-y-1">
                    <p className="text-sm text-muted-foreground">No endpoints configured.</p>
                    <p className="text-xs text-muted-foreground">
                      Add an endpoint in the MLflow AI Gateway to connect an LLM for SDG and evaluation jobs.
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
