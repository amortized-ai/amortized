import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router"
import SettingsPage from "../page"
import type { HealthResponse, ConfigResponse, MlflowGatewayRoute } from "@/types/api"

const mockGetHealth = vi.fn()
const mockGetConfig = vi.fn()
const mockGetGatewayRoutes = vi.fn()
const mockSearchMlflowRuns = vi.fn()

vi.mock("@/lib/api-client", () => ({
  getHealth: (...args: unknown[]) => mockGetHealth(...args),
  getConfig: (...args: unknown[]) => mockGetConfig(...args),
  getMlflowGatewayRoutes: (...args: unknown[]) => mockGetGatewayRoutes(...args),
  searchMlflowRuns: (...args: unknown[]) => mockSearchMlflowRuns(...args),
  getBaseUrl: () => "",
}))

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  )
}

const healthOk: HealthResponse = {
  status: "ok",
  timestamp: "2026-06-10T12:00:00Z",
}

const config: ConfigResponse = {
  mlflow_tracking_uri: "http://mlflow:5000",
  mlflow_gateway_uri: "http://mlflow:5001",
  default_compute_backend: "kubernetes",
  compute_namespace: "amortized-jobs",
  image_registry: "ghcr.io/amortized-ai",
  available_backends: ["kubernetes"],
  version: "0.3.0",
}

const routes: MlflowGatewayRoute[] = [
  {
    name: "gpt-4o",
    route_type: "llm/v1/chat",
    model: { name: "gpt-4o", provider: "openai" },
  },
  {
    name: "claude-sonnet",
    route_type: "llm/v1/chat",
    model: { name: "claude-sonnet-4-20250514", provider: "anthropic" },
  },
]

describe("SettingsPage", () => {
  it("renders the settings page with config", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument()
    expect(screen.getByText("System")).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText("http://mlflow:5000")).toBeInTheDocument()
    })
    // AI Gateway section renders once config (with a gateway URI) has loaded
    expect(screen.getAllByText("AI Gateway").length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText("kubernetes").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("0.3.0")).toBeInTheDocument()
  })

  it("renders prerequisites card", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    expect(screen.getByTestId("prerequisites-card")).toBeInTheDocument()
  })

  it("shows gateway routes when available", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGatewayRoutes.mockResolvedValue({ routes })
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    await waitFor(() => {
      expect(screen.getByText("openai/gpt-4o")).toBeInTheDocument()
    })
    expect(screen.getByText("anthropic/claude-sonnet-4-20250514")).toBeInTheDocument()
  })

  it("shows empty gateway message when no routes", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    await waitFor(() => {
      expect(screen.getByText(/No endpoints configured/)).toBeInTheDocument()
    })
  })

  it("hides the AI Gateway section when no gateway is configured (enterprise-MLflow)", async () => {
    mockGetGatewayRoutes.mockClear()
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue({ ...config, mlflow_gateway_uri: null })
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    // Platform Configuration still renders (gateway-independent)...
    await waitFor(() => {
      expect(screen.getByText("http://mlflow:5000")).toBeInTheDocument()
    })
    // ...but the vestigial AI Gateway section + its external "Add Endpoint" link are gone.
    expect(screen.queryByTestId("add-gateway-route")).not.toBeInTheDocument()
    expect(screen.queryByText("AI Gateway")).not.toBeInTheDocument()
    // ...and the gateway-routes fetch is skipped (no 404 against an unserved gateway).
    expect(mockGetGatewayRoutes).not.toHaveBeenCalled()
  })
})
