import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router"
import { describe, expect, it, vi } from "vitest"
import type { EvalResults } from "@/lib/api-client"
import type { Job } from "@/types/api"
import { JobDetailPanel } from "./job-detail-panel"

const evalResults = vi.hoisted(() => ({ current: null as EvalResults | null }))

vi.mock("../api/use-jobs", () => {
  const mutation = () => ({ mutate: vi.fn(), isPending: false })
  return {
    useCancelJob: mutation,
    useDeleteJob: mutation,
    useRetryJob: mutation,
    useJobLogs: () => ({ data: undefined, isLoading: false }),
    useJobMlflowMetrics: () => ({ data: undefined, isLoading: false }),
    useEvalResults: () => ({ data: evalResults.current, isLoading: false }),
  }
})
vi.mock("@/features/datasets/api/use-datasets", () => ({ useDatasets: () => ({ data: [] }) }))
vi.mock("@/features/models/api/use-models", () => ({ useModels: () => ({ data: [] }) }))
vi.mock("@/lib/use-resolved-config", () => ({ useResolvedConfig: <T,>(config: T) => config }))

const job: Job = {
  id: "eval-1",
  type: "eval",
  status: "succeeded",
  config: {},
  metadata: {},
  recipe: "",
  user_id: null,
  k8s_job_name: null,
  k8s_namespace: null,
  mlflow_run_id: null,
  mlflow_experiment: null,
  parent_job_id: null,
  error: null,
  created_at: "2026-11-18T09:00:00Z",
  started_at: null,
  completed_at: null,
}

async function scoreTable(results: EvalResults) {
  evalResults.current = results
  render(
    <MemoryRouter>
      <JobDetailPanel job={job} open onOpenChange={() => {}} />
    </MemoryRouter>,
  )
  await userEvent.setup().click(screen.getByRole("tab", { name: "Results" }))
  return screen.getByRole("columnheader", { name: "Criterion" }).closest("table")!
}

const base = { num_records: 10, num_skipped: 0, model: {}, scores: { match: 0.8 }, scores_n: { match: 10 } }

describe("EvalResultsTab score table", () => {
  it("has no Invalid column for results without metric metadata", async () => {
    const table = await scoreTable(base)

    expect(within(table).queryByRole("columnheader", { name: "Invalid" })).toBeNull()
    expect(within(table).getAllByRole("cell")).toHaveLength(3)
  })

  it("shows invalid counts when results carry metric metadata", async () => {
    const table = await scoreTable({
      ...base,
      metric_metadata: { match: { unit: "fraction", direction: "higher", num_invalid: 2 } },
    })

    expect(within(table).getByRole("columnheader", { name: "Invalid" })).toBeInTheDocument()
    expect(within(table).getAllByRole("cell").at(-1)).toHaveTextContent("2")
  })
})
