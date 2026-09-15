import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { EvaluationTable } from "./evaluation-table"
import { EvaluationDetailPanel } from "./evaluation-detail-panel"
import type { EvaluationGroup, EvaluationEntry } from "@/lib/api-client"
import type { ModelRecord } from "@/types/api"

vi.mock("@/features/models/api/use-models", () => ({
  useModels: () => ({
    data: [
      {
        name: "base-osft-1",
        version: "1",
        run_id: "run-1",
        source: "runs:/run-1/model",
        created_at: 0,
        description: "",
        aliases: [],
        tags: { model_display_name: "base-model" },
      } satisfies ModelRecord,
    ],
  }),
}))

function makeEntry(overrides: Partial<EvaluationEntry> = {}): EvaluationEntry {
  return {
    job_id: "11111111-2222-3333-4444-555555555555",
    model: "base-model",
    status: "succeeded",
    created_at: "2026-09-10T00:00:00Z",
    mlflow_run_id: "run-1",
    scores: { exact_match: 0.5 },
    num_samples: 100,
    topic: "test",
    ...overrides,
  }
}

function makeGroup(overrides: Partial<EvaluationGroup> = {}): EvaluationGroup {
  return {
    id: "g1",
    dataset: { run_id: "ds-run", name: "Eval Dataset" },
    metric_names: ["exact_match"],
    evals: [makeEntry()],
    latest_created_at: "2026-09-10T00:00:00Z",
    ...overrides,
  }
}

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

describe("EvaluationTable", () => {
  it("renders evaluation group rows", () => {
    const groups = [makeGroup(), makeGroup({ id: "g2", dataset: { run_id: "r2", name: "Other" } })]
    render(
      <EvaluationTable groups={groups} page={0} onPageChange={vi.fn()} onSelectGroup={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByText("Eval Dataset")).toBeInTheDocument()
    expect(screen.getByText("Other")).toBeInTheDocument()
  })

  it("shows empty state when no evaluations", () => {
    render(
      <EvaluationTable groups={[]} page={0} onPageChange={vi.fn()} onSelectGroup={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByText("No evaluations yet")).toBeInTheDocument()
  })

  it("calls onSelectGroup on row click", () => {
    const onSelect = vi.fn()
    render(
      <EvaluationTable groups={[makeGroup()]} page={0} onPageChange={vi.fn()} onSelectGroup={onSelect} />,
      { wrapper },
    )
    fireEvent.click(screen.getByText("Eval Dataset"))
    expect(onSelect).toHaveBeenCalled()
  })
})

describe("EvaluationDetailPanel", () => {
  it("renders the model comparison table with scores (one row per model)", () => {
    const group = makeGroup({
      evals: [
        makeEntry({ model: "tuned-model", scores: { accuracy: 0.9 }, job_id: "aaaaaaaa-1111" }),
        makeEntry({ model: "base-model", scores: { accuracy: 0.6 }, job_id: "bbbbbbbb-2222" }),
      ],
      metric_names: ["accuracy"],
    })
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByText("tuned-model")).toBeInTheDocument()
    expect(screen.getByText("base-model")).toBeInTheDocument()
    expect(screen.getByText("90%")).toBeInTheDocument()
    expect(screen.getByText("60%")).toBeInTheDocument()
    // metric is a row, models are column headers
    expect(screen.getByText("accuracy")).toBeInTheDocument()
  })

  it("links each model column to its job", () => {
    const group = makeGroup()
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    const link = screen.getByTestId("eval-job-link-11111111-2222-3333-4444-555555555555")
    expect(link).toBeInTheDocument()
  })

  it("shows the dataset info card with a link to the Datasets tab", () => {
    const group = makeGroup()
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByText("Dataset", { selector: "span" })).toBeInTheDocument()
    expect(screen.getByTestId("eval-dataset-link")).toHaveTextContent("Eval Dataset")
  })

  it("links model names to the Models tab", () => {
    const group = makeGroup()
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByTestId("eval-model-link-base-model")).toBeInTheDocument()
  })

  it("merges duplicate model evals into one column with mean ± std", () => {
    const group = makeGroup({
      evals: [
        makeEntry({ model: "tuned-model", scores: { accuracy: 0.8 }, job_id: "aaaaaaaa-1111" }),
        makeEntry({ model: "tuned-model", scores: { accuracy: 0.6 }, job_id: "bbbbbbbb-2222" }),
        makeEntry({ model: "base-model", scores: { accuracy: 0.4 }, job_id: "cccccccc-3333" }),
      ],
      metric_names: ["accuracy"],
    })
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    // one column per model, not per run
    expect(screen.getAllByText("tuned-model").length).toBe(1)
    // mean of 0.8/0.6 = 0.7, sample std of two values = 0.1
    expect(screen.getByText("70%")).toBeInTheDocument()
    expect(screen.getByText("±14.1%")).toBeInTheDocument()
    // base model single run shows its own score without std
    expect(screen.getByText("40%")).toBeInTheDocument()
    expect(screen.queryByText("±0%")).not.toBeInTheDocument()
    // both job links of the merged runs remain
    expect(screen.getByTestId("eval-job-link-aaaaaaaa-1111")).toBeInTheDocument()
    expect(screen.getByTestId("eval-job-link-bbbbbbbb-2222")).toBeInTheDocument()
  })

  it("renders unresolvable models as plain text, not links", () => {
    const group = makeGroup({
      evals: [makeEntry({ model: "some/hf-base-model" })],
    })
    render(
      <EvaluationDetailPanel group={group} open={true} onOpenChange={vi.fn()} />,
      { wrapper },
    )
    expect(screen.getByText("some/hf-base-model")).toBeInTheDocument()
    expect(
      screen.queryByTestId("eval-model-link-some/hf-base-model"),
    ).not.toBeInTheDocument()
  })
})
