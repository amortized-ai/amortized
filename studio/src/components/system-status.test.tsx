import { render, screen } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { SystemStatus } from "@/components/system-status"

vi.mock("@/hooks/use-system-health", () => ({
  useSystemHealth: vi.fn(),
}))

import { useSystemHealth } from "@/hooks/use-system-health"

const mockUseSystemHealth = vi.mocked(useSystemHealth)

describe("SystemStatus", () => {
  it("renders backend status badge", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "ok", mlflow: "ok" })

    render(<SystemStatus />)

    expect(screen.getByTestId("system-status")).toBeInTheDocument()
    expect(screen.getByTestId("status-backend")).toBeInTheDocument()
  })

  it("shows green badge when backend is ok", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "ok", mlflow: "ok" })

    render(<SystemStatus />)

    const badge = screen.getByTestId("status-backend")
    expect(badge.className).toContain("bg-rh-green-light")
    expect(badge).toHaveTextContent("Ready")
  })

  it("shows red badge for error status", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "error", mlflow: "ok" })

    render(<SystemStatus />)

    const badge = screen.getByTestId("status-backend")
    expect(badge.className).toContain("bg-rh-danger-light")
    expect(badge).toHaveTextContent("Unavailable")
  })

  it("shows loading state with Checking text", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "loading", mlflow: "loading" })

    render(<SystemStatus />)

    const badge = screen.getByTestId("status-backend")
    expect(badge).toHaveTextContent("Checking…")
  })

  it("displays Backend label", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "ok", mlflow: "ok" })

    render(<SystemStatus />)

    expect(screen.getByText("Backend")).toBeInTheDocument()
  })

  it("renders mlflow status badge", () => {
    mockUseSystemHealth.mockReturnValue({ backend: "ok", mlflow: "error" })

    render(<SystemStatus />)

    const badge = screen.getByTestId("status-mlflow")
    expect(badge).toBeInTheDocument()
    expect(badge).toHaveTextContent("Unavailable")
  })
})
