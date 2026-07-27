import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router"
import { PrerequisitesCard } from "./prerequisites-card"
import type { HealthResponse } from "@/types/api"

const mockGetHealth = vi.fn()
const mockSearchMlflowRuns = vi.fn()

vi.mock("@/lib/api-client", () => ({
  getHealth: (...args: unknown[]) => mockGetHealth(...args),
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

describe("PrerequisitesCard", () => {
  it("renders the card with 2 checks (backend and mlflow)", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<PrerequisitesCard />, { wrapper })

    expect(screen.getByTestId("prerequisites-card")).toBeInTheDocument()
    expect(screen.getByText("Service Connections")).toBeInTheDocument()
    expect(screen.getByText("Backend Connection")).toBeInTheDocument()
    expect(screen.getByText("MLflow Connection")).toBeInTheDocument()
  })

  it("shows 'All services connected' summary when both checks pass", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<PrerequisitesCard />, { wrapper })

    expect(await screen.findByText("All services connected")).toBeInTheDocument()
  })

  it("shows error when backend connection fails", async () => {
    mockGetHealth.mockRejectedValue(new Error("Network error"))
    mockSearchMlflowRuns.mockResolvedValue({ runs: [] })

    render(<PrerequisitesCard />, { wrapper })

    await waitFor(() => {
      expect(screen.getByTestId("prereq-badge-backend-connection")).toHaveTextContent("Unavailable")
    })
  })

  it("shows error when mlflow connection fails", async () => {
    mockGetHealth.mockResolvedValue(healthOk)
    mockSearchMlflowRuns.mockRejectedValue(new Error("MLflow unreachable"))

    render(<PrerequisitesCard />, { wrapper })

    await waitFor(() => {
      expect(screen.getByTestId("prereq-badge-mlflow-connection")).toHaveTextContent("Unavailable")
    })
  })

  it("shows error when both fail", async () => {
    mockGetHealth.mockRejectedValue(new Error("Network error"))
    mockSearchMlflowRuns.mockRejectedValue(new Error("MLflow unreachable"))

    render(<PrerequisitesCard />, { wrapper })

    await waitFor(() => {
      expect(screen.getByTestId("prereq-badge-backend-connection")).toHaveTextContent("Unavailable")
    })
    expect(screen.getByTestId("prereq-badge-mlflow-connection")).toHaveTextContent("Unavailable")
  })
})
