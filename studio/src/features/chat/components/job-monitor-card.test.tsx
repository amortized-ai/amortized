import { render, screen, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { JobMonitorCard } from "./job-monitor-card"

const getJob = vi.fn()
const getEvalEndpointSuggestions = vi.fn()

vi.mock("@/lib/api-client", () => ({
  getJob: (...args: unknown[]) => getJob(...args),
  getEvalEndpointSuggestions: (...args: unknown[]) => getEvalEndpointSuggestions(...args),
}))

function job(status: string) {
  return {
    id: "9f226235-e8bd-433c-aca0-4a42181210f4",
    status,
    created_at: "2026-09-11T15:00:00Z",
    started_at: "2026-09-11T15:00:01Z",
    mlflow_run_id: "",
    type: "serve",
    config: {},
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  getJob.mockResolvedValue(job("running"))
  getEvalEndpointSuggestions.mockResolvedValue({
    serve_endpoints: [
      {
        job_id: "9f226235-e8bd-433c-aca0-4a42181210f4",
        name: "m",
        model_name: "m",
        base_url: "http://x/v1",
        healthy: true,
      },
    ],
  })
})

describe("JobMonitorCard — serve jobs", () => {
  it("ends monitoring when the endpoint becomes healthy", async () => {
    const onComplete = vi.fn()
    render(
      <JobMonitorCard
        jobId="9f226235-e8bd-433c-aca0-4a42181210f4"
        jobType="SERVE"
        onComplete={onComplete}
      />,
    )
    await waitFor(
      () => expect(screen.getByText("Endpoint ready")).toBeInTheDocument(),
      { timeout: 3000 },
    )
    expect(onComplete).toHaveBeenCalledWith(
      "9f226235-e8bd-433c-aca0-4a42181210f4",
      "SERVE",
      "succeeded",
    )
  })

  it("keeps monitoring while the serve endpoint is not healthy", async () => {
    getEvalEndpointSuggestions.mockResolvedValue({
      serve_endpoints: [
        {
          job_id: "9f226235-e8bd-433c-aca0-4a42181210f4",
          name: "m",
          model_name: "m",
          base_url: "http://x/v1",
          healthy: false,
        },
      ],
    })
    render(
      <JobMonitorCard
        jobId="9f226235-e8bd-433c-aca0-4a42181210f4"
        jobType="SERVE"
      />,
    )
    await waitFor(() => expect(getJob).toHaveBeenCalled())
    expect(screen.queryByText("Endpoint ready")).not.toBeInTheDocument()
  })

  it("fires completion with the job status when the serve job is cancelled", async () => {
    getJob.mockResolvedValue(job("cancelled"))
    const onComplete = vi.fn()
    render(
      <JobMonitorCard
        jobId="9f226235-e8bd-433c-aca0-4a42181210f4"
        jobType="SERVE"
        onComplete={onComplete}
      />,
    )
    await waitFor(
      () => expect(onComplete).toHaveBeenCalledWith(
        "9f226235-e8bd-433c-aca0-4a42181210f4",
        "SERVE",
        "cancelled",
      ),
      { timeout: 3000 },
    )
  })
})
