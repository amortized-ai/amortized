import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"

vi.mock("@/features/settings/api/use-settings", () => ({
  useHealth: () => ({ data: { status: "ok", gpu: null }, isError: false, isLoading: false }),
}))

vi.mock("@/features/jobs/api/use-jobs", () => ({
  useJobs: () => ({ data: [], isLoading: false }),
}))

vi.mock("@/features/datasets/api/use-datasets", () => ({
  useDatasets: () => ({ data: [], isLoading: false }),
}))

vi.mock("@/features/models/api/use-models", () => ({
  useModels: () => ({ data: [], isLoading: false }),
}))

import App from "./App"

describe("App", () => {
  it("renders without crashing", async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Amortized Studio" })).toBeInTheDocument()
    })
  })

  it("renders the overview page by default", async () => {
    render(<App />)
    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Amortized Studio" }),
      ).toBeInTheDocument()
    })
  })
})
