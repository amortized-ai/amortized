import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { DatasetTable } from "./dataset-table"
import { DatasetOverviewTab } from "./dataset-overview-tab"
import { DatasetLineage } from "./dataset-lineage"
import type { DatasetRecord } from "@/types/api"

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

function makeDataset(overrides: Partial<DatasetRecord> = {}): DatasetRecord {
  return {
    run_id: "run-1",
    name: "Training Data",
    run_name: "sdg-run-1",
    experiment_id: "exp-1",
    artifact_uri: "s3://bucket/artifacts/run-1",
    created_at: Date.now(),
    metrics: { num_samples_generated: 100 },
    params: { model: "gpt-4o" },
    tags: { dataset_name: "Training Data", job_type: "sdg", teacher_model: "gpt-4o", num_samples: "100" },
    ...overrides,
  }
}

describe("DatasetTable", () => {
  it("renders dataset rows", () => {
    const datasets = [
      makeDataset({ run_id: "r1", name: "Training Data" }),
      makeDataset({ run_id: "r2", name: "Eval Data" }),
    ]
    render(
      <DatasetTable
        datasets={datasets}
        page={0}
        onPageChange={vi.fn()}
        onSelectDataset={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("Training Data")).toBeInTheDocument()
    expect(screen.getByText("Eval Data")).toBeInTheDocument()
  })

  it("shows empty state when no datasets", () => {
    render(
      <DatasetTable
        datasets={[]}
        page={0}
        onPageChange={vi.fn()}
        onSelectDataset={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("No datasets yet")).toBeInTheDocument()
  })

  it("calls onSelectDataset when row clicked", () => {
    const onSelect = vi.fn()
    const ds = makeDataset({ run_id: "r1" })
    render(
      <DatasetTable
        datasets={[ds]}
        page={0}
        onPageChange={vi.fn()}
        onSelectDataset={onSelect}
      />,
      { wrapper: Wrapper },
    )
    fireEvent.click(screen.getByTestId("dataset-row-r1"))
    expect(onSelect).toHaveBeenCalledWith(ds)
  })

  it("renders pagination when more than 10 datasets", () => {
    const datasets = Array.from({ length: 15 }, (_, i) =>
      makeDataset({ run_id: `r${i}`, name: `Dataset ${i}` }),
    )
    render(
      <DatasetTable
        datasets={datasets}
        page={0}
        onPageChange={vi.fn()}
        onSelectDataset={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument()
  })

  it("displays model and samples columns", () => {
    render(
      <DatasetTable
        datasets={[makeDataset()]}
        page={0}
        onPageChange={vi.fn()}
        onSelectDataset={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("gpt-4o")).toBeInTheDocument()
    expect(screen.getByText("100")).toBeInTheDocument()
  })
})

describe("DatasetOverviewTab", () => {
  it("renders overview tab with dataset info", () => {
    const dataset = makeDataset({ run_id: "run-abc" })
    render(<DatasetOverviewTab dataset={dataset} />)
    expect(screen.getByTestId("overview-tab")).toBeInTheDocument()
    expect(screen.getByText("run-abc")).toBeInTheDocument()
  })

  it("displays model and samples info", () => {
    const dataset = makeDataset({
      params: { model: "claude-sonnet" },
      metrics: { num_samples_generated: 250 },
    })
    render(<DatasetOverviewTab dataset={dataset} />)
    expect(screen.getByText("claude-sonnet")).toBeInTheDocument()
    expect(screen.getByText("250")).toBeInTheDocument()
  })
})

describe("DatasetLineage", () => {
  it("renders parent job link when parentJobId is provided", () => {
    render(<DatasetLineage parentJobId="job-123" />, { wrapper: Wrapper })
    expect(screen.getByText("Parent Job: job-123")).toBeInTheDocument()
  })

  it("renders no parent message when parentJobId is null", () => {
    render(<DatasetLineage parentJobId={null} />, { wrapper: Wrapper })
    expect(screen.getByText("No parent job linked.")).toBeInTheDocument()
  })
})
