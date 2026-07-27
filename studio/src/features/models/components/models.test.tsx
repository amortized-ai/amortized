import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ModelTable } from "./model-table"
import { ModelDetail } from "./model-detail"
import { TrainingCharts } from "./training-charts"
import type { ModelRecord, TrainingMetric } from "@/types/api"

vi.mock("../api/use-models", () => ({
  useModelRunData: vi.fn(() => ({
    data: null,
    isLoading: false,
  })),
  useModelJobs: vi.fn(() => ({
    data: { trainingJob: null, sdgJob: null, evalJobs: [] },
    isLoading: false,
  })),
}))

function makeModel(overrides: Partial<ModelRecord> = {}): ModelRecord {
  return {
    name: "my-fine-tuned-model",
    version: "1",
    run_id: "run-abc",
    source: "s3://bucket/models/my-fine-tuned-model",
    created_at: Date.now(),
    description: "A fine-tuned model",
    aliases: ["champion"],
    tags: {},
    ...overrides,
  }
}

function makeMetrics(): TrainingMetric[] {
  return [
    { step: 1, loss: 2.5, learning_rate: 0.001, grad_norm: 1.2, epoch: 0, max_steps: 3 },
    { step: 2, loss: 2.0, learning_rate: 0.001, grad_norm: 1.0, epoch: 0, max_steps: 3 },
    { step: 3, loss: 1.5, learning_rate: 0.001, grad_norm: 0.8, epoch: 1, max_steps: 3 },
  ]
}

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

describe("ModelTable", () => {
  it("renders model rows with correct columns", () => {
    const models = [
      makeModel({ name: "model-a", version: "1" }),
      makeModel({ name: "model-b", version: "2", aliases: [] }),
    ]
    render(
      <MemoryRouter>
        <ModelTable
          models={models}
          page={0}
          onPageChange={vi.fn()}
          onSelectModel={vi.fn()}
        />
      </MemoryRouter>,
    )
    expect(screen.getByText("model-a")).toBeInTheDocument()
    expect(screen.getByText("model-b")).toBeInTheDocument()
    expect(screen.getByText("v1")).toBeInTheDocument()
    expect(screen.getByText("v2")).toBeInTheDocument()
  })

  it("shows empty state when no models", () => {
    render(
      <MemoryRouter>
        <ModelTable
          models={[]}
          page={0}
          onPageChange={vi.fn()}
          onSelectModel={vi.fn()}
        />
      </MemoryRouter>,
    )
    expect(screen.getByText("No models yet")).toBeInTheDocument()
  })

  it("calls onSelectModel when row clicked", () => {
    const onSelect = vi.fn()
    const model = makeModel({ name: "click-me" })
    render(
      <MemoryRouter>
        <ModelTable
          models={[model]}
          page={0}
          onPageChange={vi.fn()}
          onSelectModel={onSelect}
        />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByTestId("model-row-click-me"))
    expect(onSelect).toHaveBeenCalledWith(model)
  })

  it("shows aliases column", () => {
    render(
      <MemoryRouter>
        <ModelTable
          models={[makeModel({ aliases: ["champion", "latest"] })]}
          page={0}
          onPageChange={vi.fn()}
          onSelectModel={vi.fn()}
        />
      </MemoryRouter>,
    )
    expect(screen.getByText("champion, latest")).toBeInTheDocument()
  })

  it("renders pagination when more than 10 models", () => {
    const models = Array.from({ length: 15 }, (_, i) =>
      makeModel({ name: `model-${i}` }),
    )
    render(
      <MemoryRouter>
        <ModelTable
          models={models}
          page={0}
          onPageChange={vi.fn()}
          onSelectModel={vi.fn()}
        />
      </MemoryRouter>,
    )
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument()
  })
})

describe("ModelDetail", () => {
  it("renders model name and version info", () => {
    render(
      <ModelDetail
        name="my-fine-tuned-model"
        versions={[makeModel()]}
        onBack={vi.fn()}
      />,
      { wrapper },
    )
    expect(screen.getByText("my-fine-tuned-model")).toBeInTheDocument()
    expect(screen.getByText("@champion")).toBeInTheDocument()
  })

  it("calls onBack when back button clicked", () => {
    const onBack = vi.fn()
    render(
      <ModelDetail
        name="my-fine-tuned-model"
        versions={[makeModel()]}
        onBack={onBack}
      />,
      { wrapper },
    )
    fireEvent.click(screen.getByText("Back"))
    expect(onBack).toHaveBeenCalled()
  })

  it("shows MLflow link when run_id is present", () => {
    render(
      <ModelDetail
        name="my-fine-tuned-model"
        versions={[makeModel({ run_id: "run-xyz" })]}
        onBack={vi.fn()}
      />,
      { wrapper },
    )
    expect(screen.getByText("View in MLflow")).toBeInTheDocument()
  })
})

describe("TrainingCharts", () => {
  it("renders loss chart from metrics data", () => {
    render(<TrainingCharts metrics={makeMetrics()} isLoading={false} />)
    expect(screen.getByText("Training Loss")).toBeInTheDocument()
    expect(screen.queryByText("No training metrics available.")).not.toBeInTheDocument()
  })

  it("renders gradient norm chart when data present", () => {
    render(<TrainingCharts metrics={makeMetrics()} isLoading={false} />)
    expect(screen.getByText("Gradient Norm")).toBeInTheDocument()
  })

  it("hides gradient norm chart when no grad_norm data", () => {
    const metrics = makeMetrics().map((m) => ({
      ...m,
      grad_norm: null,
    }))
    render(<TrainingCharts metrics={metrics} isLoading={false} />)
    expect(screen.getByText("Training Loss")).toBeInTheDocument()
    expect(screen.queryByText("Gradient Norm")).not.toBeInTheDocument()
  })

  it("shows empty state when no metrics", () => {
    render(<TrainingCharts metrics={[]} isLoading={false} />)
    expect(
      screen.getByText("No training metrics available."),
    ).toBeInTheDocument()
  })
})
