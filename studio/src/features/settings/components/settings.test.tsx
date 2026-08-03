import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router"
import SettingsPage from "../page"
import type { HealthResponse, ConfigResponse, MlflowGatewayRoute } from "@/types/api"

const mockGetHealth = vi.fn()
const mockGetConfig = vi.fn()
const mockGetGpuUtilization = vi.fn()
const mockGetGatewayRoutes = vi.fn()
const mockGetGatewayConnections = vi.fn()
const mockSearchMlflowRuns = vi.fn()

vi.mock("@/lib/api-client", () => ({
  getHealth: (...args: unknown[]) => mockGetHealth(...args),
  getConfig: (...args: unknown[]) => mockGetConfig(...args),
  getGpuUtilization: (...args: unknown[]) => mockGetGpuUtilization(...args),
  getMlflowGatewayRoutes: (...args: unknown[]) => mockGetGatewayRoutes(...args),
  getMlflowGatewayConnections: (...args: unknown[]) => mockGetGatewayConnections(...args),
  createMlflowGatewayRoute: vi.fn(),
  deleteMlflowGatewayRoute: vi.fn(),
  createMlflowGatewayConnection: vi.fn(),
  deleteMlflowGatewayConnection: vi.fn(),
  searchMlflowRuns: (...args: unknown[]) => mockSearchMlflowRuns(...args),
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
    mockGetGpuUtilization.mockResolvedValue({ available: false, total_gpus_in_use: 0, jobs: [], reason: "requires_kubernetes_backend" })
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockGetGatewayConnections.mockResolvedValue([])
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument()
    expect(screen.getByText("System")).toBeInTheDocument()
    expect(screen.getAllByText("AI Gateway").length).toBeGreaterThanOrEqual(1)

    await waitFor(() => {
      expect(screen.getByText("http://mlflow:5000")).toBeInTheDocument()
    })
    expect(screen.getAllByText("kubernetes").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("0.3.0")).toBeInTheDocument()
  })

  it("renders prerequisites card", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGpuUtilization.mockResolvedValue({ available: false, total_gpus_in_use: 0, jobs: [], reason: "requires_kubernetes_backend" })
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockGetGatewayConnections.mockResolvedValue([])
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    expect(screen.getByTestId("prerequisites-card")).toBeInTheDocument()
  })

  it("shows gateway routes when available", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockGetConfig.mockResolvedValue(config)
    mockGetGpuUtilization.mockResolvedValue({ available: false, total_gpus_in_use: 0, jobs: [], reason: "requires_kubernetes_backend" })
    mockGetGatewayRoutes.mockResolvedValue({ routes })
    mockGetGatewayConnections.mockResolvedValue([])
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
    mockGetGpuUtilization.mockResolvedValue({ available: false, total_gpus_in_use: 0, jobs: [], reason: "requires_kubernetes_backend" })
    mockGetGatewayRoutes.mockResolvedValue({ routes: [] })
    mockGetGatewayConnections.mockResolvedValue([])
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<SettingsPage />, { wrapper })

    await waitFor(() => {
      expect(screen.getByText(/No endpoints configured/)).toBeInTheDocument()
    })
  })
})
