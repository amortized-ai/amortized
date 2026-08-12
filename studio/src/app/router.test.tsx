import { render, screen } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter, Routes, Route, Navigate } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { TooltipProvider } from "@/components/ui/tooltip"
import { AppLayout } from "@/app/layout"
import OverviewPage from "@/features/overview/page"
import ChatPage from "@/features/chat/page"
import JobsPage from "@/features/jobs/page"
import DatasetsPage from "@/features/datasets/page"
import { DatasetDetailPage } from "@/features/datasets/components/dataset-detail"
import ModelsPage from "@/features/models/page"
import SettingsPage from "@/features/settings/page"

vi.mock("@/features/settings/api/use-settings", () => ({
  useHealth: () => ({ data: { status: "ok" }, isError: false, isLoading: false }),
  useConfig: () => ({ data: undefined, isLoading: false }),
  useGatewayRoutes: () => ({ data: [], isLoading: false }),
}))

vi.mock("@/features/jobs/api/use-jobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/jobs/api/use-jobs")>()),
  useJobs: () => ({ data: [], isLoading: false }),
}))

vi.mock("@/features/datasets/api/use-datasets", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/datasets/api/use-datasets")>()),
  useDatasets: () => ({ data: [], isLoading: false }),
  useDataset: () => ({ data: null, isLoading: false }),
}))

vi.mock("@/features/models/api/use-models", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/models/api/use-models")>()),
  useModels: () => ({ data: [], isLoading: false }),
  useModel: () => ({ data: null, isLoading: false }),
  useModelMetrics: () => ({ data: [], isLoading: false }),
}))

function renderRoute(initialRoute: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[initialRoute]}>
          <Routes>
            <Route element={<AppLayout />}>
              <Route index element={<Navigate to="/overview" replace />} />
              <Route path="overview" element={<OverviewPage />} />
              <Route path="chat" element={<ChatPage />} />
              <Route path="jobs" element={<JobsPage />} />
              <Route path="datasets" element={<DatasetsPage />} />
              <Route path="datasets/:id" element={<DatasetDetailPage />} />
              <Route path="models" element={<ModelsPage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )
}

describe("Router", () => {
  it("redirects / to /overview", () => {
    renderRoute("/")
    expect(screen.getByRole("heading", { name: "Amortized Studio" })).toBeInTheDocument()
  })

  it("renders the chat page at /chat", () => {
    renderRoute("/chat")
    expect(screen.getByPlaceholderText("Type a message...")).toBeInTheDocument()
  })

  it("renders the jobs page at /jobs", () => {
    renderRoute("/jobs")
    expect(screen.getByRole("heading", { name: "Jobs" })).toBeInTheDocument()
  })

  it("renders the datasets page at /datasets", () => {
    renderRoute("/datasets")
    expect(screen.getByRole("heading", { name: "Datasets" })).toBeInTheDocument()
  })

  it("renders the models page at /models", () => {
    renderRoute("/models")
    expect(screen.getByRole("heading", { name: "Models" })).toBeInTheDocument()
  })

  it("renders the settings page at /settings", () => {
    renderRoute("/settings")
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument()
  })

  it("renders the dataset detail page at /datasets/:id", () => {
    renderRoute("/datasets/ds-123")
    expect(screen.getByText("Dataset not found.")).toBeInTheDocument()
  })

  it("renders the models page at /models with query params", () => {
    renderRoute("/models?run=test-run-id")
    expect(screen.getByRole("heading", { name: "Models" })).toBeInTheDocument()
  })
})
