import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { JobMonitorCard } from "./job-monitor-card"

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
