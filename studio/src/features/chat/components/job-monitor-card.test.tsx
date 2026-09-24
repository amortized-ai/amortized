import { render as rtlRender, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { MemoryRouter } from "react-router"
import { JobMonitorCard } from "./job-monitor-card"

// JobMonitorCard renders react-router <Link>s in its success state. Render under a NON-ROOT
// basename so the tests can assert the success links actually route through it — that's the
// point of the fix (an absolute <a href> would bypass the basename and 404 in the embed, and
// would still pass a test that only provided router context).
const BASENAME = "/amortized-studio-embed"
function render(ui: React.ReactElement) {
  return rtlRender(
    <MemoryRouter basename={BASENAME} initialEntries={[`${BASENAME}/`]}>
      {ui}
    </MemoryRouter>,
  )
}

const getJob = vi.fn()
const getJobLogs = vi.fn()

vi.mock("@/lib/api-client", () => ({
  getJob: (...args: unknown[]) => getJob(...args),
  getJobLogs: (...args: unknown[]) => getJobLogs(...args),
  getJobDurationStats: () => Promise.resolve({}),
}))

function job(status: string, type = "eval") {
  return {
    id: "34bd1852-e8bd-433c-aca0-4a42181210f4",
    status,
    created_at: "2026-09-11T15:00:00Z",
    started_at: "2026-09-11T15:00:01Z",
    mlflow_run_id: "run-1",
    type,
    config: {},
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  getJob.mockResolvedValue(job("running"))
  getJobLogs.mockResolvedValue(["=== EVAL-STAGE: serving ==="])
})

describe("JobMonitorCard — eval jobs with embedded serving", () => {
  it("shows the serve stage from log markers while the model loads", async () => {
    render(
      <JobMonitorCard jobId="34bd1852-e8bd-433c-aca0-4a42181210f4" jobType="EVAL" />,
    )
    await waitFor(
      () =>
        expect(
          screen.getByText("Serving the model inside the eval job (Stage 3/4)"),
        ).toBeInTheDocument(),
      { timeout: 3000 },
    )
  })

  it("shows the evaluating stage once the marker advances", async () => {
    getJobLogs.mockResolvedValue([
      "=== EVAL-STAGE: serving ===",
      "=== EVAL-STAGE: waiting-for-endpoint ===",
      "=== EVAL-STAGE: evaluating ===",
    ])
    render(
      <JobMonitorCard jobId="34bd1852-e8bd-433c-aca0-4a42181210f4" jobType="EVAL" />,
    )
    await waitFor(
      () =>
        expect(screen.getByText("Evaluating the model (Stage 3/4)")).toBeInTheDocument(),
      { timeout: 3000 },
    )
  })

  it("fires completion with the job status when the eval job succeeds", async () => {
    getJob.mockResolvedValue(job("succeeded"))
    const onComplete = vi.fn()
    render(
      <JobMonitorCard
        jobId="34bd1852-e8bd-433c-aca0-4a42181210f4"
        jobType="EVAL"
        onComplete={onComplete}
      />,
    )
    await waitFor(
      () =>
        expect(onComplete).toHaveBeenCalledWith(
          "34bd1852-e8bd-433c-aca0-4a42181210f4",
          "EVAL",
          "succeeded",
        ),
      { timeout: 3000 },
    )
  })

  it("does not fetch logs for non-eval jobs", async () => {
    getJob.mockResolvedValue(job("running", "training"))
    render(
      <JobMonitorCard jobId="34bd1852-e8bd-433c-aca0-4a42181210f4" jobType="TRAINING" />,
    )
    await waitFor(() => expect(getJob).toHaveBeenCalled())
    expect(getJobLogs).not.toHaveBeenCalled()
  })
})

describe("JobMonitorCard — dataset splits", () => {
  it("shows the split stage label while the split runs", async () => {
    getJob.mockResolvedValue(job("running", "upload"))
    render(
      <JobMonitorCard jobId="34bd1852-e8bd-433c-aca0-4a42181210f4" jobType="SPLIT" />,
    )
    await waitFor(
      () => expect(screen.getByText("Splitting dataset (Stage 3/4)")).toBeInTheDocument(),
      { timeout: 3000 },
    )
  })

  it("fires completion with the job status when the split succeeds", async () => {
    getJob.mockResolvedValue(job("succeeded", "upload"))
    const onComplete = vi.fn()
    render(
      <JobMonitorCard
        jobId="34bd1852-e8bd-433c-aca0-4a42181210f4"
        jobType="SPLIT"
        onComplete={onComplete}
      />,
    )
    await waitFor(() =>
      expect(onComplete).toHaveBeenCalledWith(
        "34bd1852-e8bd-433c-aca0-4a42181210f4",
        "SPLIT",
        "succeeded",
      ),
    )
  })
})

// The point of the fix: success links must route through the router basename (react-router
// <Link>), not absolute <a href> that bypass it and 404 in the embed. Assert each rendered
// href is basename-prefixed so a regression to absolute anchors fails.
describe("JobMonitorCard — success links route through the basename", () => {
  const jobId = "34bd1852-e8bd-433c-aca0-4a42181210f4"

  it("View Job (shared across job types) is basename-prefixed", async () => {
    getJob.mockResolvedValue(job("succeeded", "eval"))
    render(<JobMonitorCard jobId={jobId} jobType="EVAL" />)
    const link = await screen.findByRole("link", { name: /View Job/ }, { timeout: 3000 })
    expect(link.getAttribute("href")).toContain(`${BASENAME}/jobs`)
  })

  it("TRAINING → View Model is basename-prefixed", async () => {
    getJob.mockResolvedValue(job("succeeded", "training"))
    render(<JobMonitorCard jobId={jobId} jobType="TRAINING" />)
    const link = await screen.findByRole("link", { name: /View Model/ }, { timeout: 3000 })
    expect(link.getAttribute("href")).toContain(`${BASENAME}/models`)
  })

  it("EVAL → View Results is basename-prefixed", async () => {
    getJob.mockResolvedValue(job("succeeded", "eval"))
    render(<JobMonitorCard jobId={jobId} jobType="EVAL" />)
    const link = await screen.findByRole("link", { name: /View Results/ }, { timeout: 3000 })
    expect(link.getAttribute("href")).toBe(`${BASENAME}/evaluation`)
  })

  it("SPLIT/SDG → View Dataset is basename-prefixed", async () => {
    getJob.mockResolvedValue(job("succeeded", "upload"))
    render(<JobMonitorCard jobId={jobId} jobType="SPLIT" />)
    const link = await screen.findByRole("link", { name: /View Dataset/ }, { timeout: 3000 })
    expect(link.getAttribute("href")).toContain(`${BASENAME}/datasets`)
  })
})
