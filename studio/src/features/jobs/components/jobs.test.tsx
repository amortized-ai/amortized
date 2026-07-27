import type React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { TooltipProvider } from "@/components/ui/tooltip"
import { StatusBadge } from "./status-badge"
import { JobTypeBadge } from "./job-type-badge"
import { FilterChips } from "./filter-chips"
import { JobTable } from "./job-table"
import type { Job, JobStatus, JobType } from "@/types/api"

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    type: "training",
    status: "running",
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
    created_at: "2026-06-10T12:00:00Z",
    started_at: "2026-06-10T12:00:00Z",
    completed_at: null,
    ...overrides,
  }
}

describe("StatusBadge", () => {
  const statuses: { status: JobStatus; label: string; classFragment: string }[] = [
    { status: "queued", label: "Queued", classFragment: "bg-muted" },
    { status: "provisioning", label: "Provisioning", classFragment: "bg-rh-blue" },
    { status: "running", label: "Running", classFragment: "bg-rh-blue" },
    { status: "succeeded", label: "Succeeded", classFragment: "bg-rh-green" },
    { status: "failed", label: "Failed", classFragment: "bg-rh-danger" },
    { status: "cancelled", label: "Cancelled", classFragment: "bg-rh-yellow" },
  ]

  statuses.forEach(({ status, label, classFragment }) => {
    it(`renders ${status} badge with correct label and styling`, () => {
      render(<StatusBadge status={status} />)
      const badge = screen.getByText(label)
      expect(badge).toBeInTheDocument()
      expect(badge.className).toContain(classFragment)
    })
  })

  it("running badge has animate-pulse", () => {
    render(<StatusBadge status="running" />)
    const badge = screen.getByText("Running")
    expect(badge.className).toContain("animate-pulse")
  })
})

describe("JobTypeBadge", () => {
  const types: { type: JobType; label: string }[] = [
    { type: "training", label: "Training" },
    { type: "sdg", label: "SDG" },
    { type: "eval", label: "Eval" },
  ]

  types.forEach(({ type, label }) => {
    it(`renders ${type} as ${label}`, () => {
      render(<JobTypeBadge type={type} />)
      expect(screen.getByText(label)).toBeInTheDocument()
    })
  })
})

describe("FilterChips", () => {
  const options = [
    { value: "training" as const, label: "Training" },
    { value: "eval" as const, label: "Eval" },
  ]

  it("renders all options", () => {
    render(
      <FilterChips label="Type" options={options} selected={[]} onChange={vi.fn()} />,
    )
    expect(screen.getByText("Training")).toBeInTheDocument()
    expect(screen.getByText("Eval")).toBeInTheDocument()
  })

  it("toggles selection on click", () => {
    const onChange = vi.fn()
    render(
      <FilterChips label="Type" options={options} selected={[]} onChange={onChange} />,
    )
    fireEvent.click(screen.getByText("Training"))
    expect(onChange).toHaveBeenCalledWith(["training"])
  })

  it("removes from selection when already selected", () => {
    const onChange = vi.fn()
    render(
      <FilterChips
        label="Type"
        options={options}
        selected={["training"]}
        onChange={onChange}
      />,
    )
    fireEvent.click(screen.getByText("Training"))
    expect(onChange).toHaveBeenCalledWith([])
  })
})

function renderJobTable(ui: React.ReactElement) {
  return render(
    <MemoryRouter>
      <TooltipProvider>{ui}</TooltipProvider>
    </MemoryRouter>,
  )
}

describe("JobTable", () => {
  it("renders job rows with correct data", () => {
    const jobs = [
      makeJob({ id: "j1", type: "training", status: "running" }),
      makeJob({ id: "j2", type: "eval", status: "succeeded" }),
    ]

    renderJobTable(
      <JobTable jobs={jobs} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )

    expect(screen.getByText("Training")).toBeInTheDocument()
    expect(screen.getByText("Eval")).toBeInTheDocument()
    expect(screen.getByText("Running")).toBeInTheDocument()
    expect(screen.getByText("Succeeded")).toBeInTheDocument()
  })

  it("shows dash for elapsed when job has no start time", () => {
    const jobs = [makeJob({ status: "queued", started_at: null })]
    renderJobTable(
      <JobTable jobs={jobs} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )
    expect(screen.getByText("—")).toBeInTheDocument()
  })

  it("renders copy button for job ID", () => {
    const job = makeJob({ id: "abcdef1234567890" })
    renderJobTable(
      <JobTable jobs={[job]} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )
    expect(screen.getByLabelText("Copy job ID abcdef1234567890")).toBeInTheDocument()
  })

  it("calls onSelectJob when row is clicked", () => {
    const onSelectJob = vi.fn()
    const job = makeJob({ id: "j1" })
    renderJobTable(
      <JobTable jobs={[job]} page={0} onPageChange={vi.fn()} onSelectJob={onSelectJob} />,
    )
    fireEvent.click(screen.getByTestId("job-row-j1"))
    expect(onSelectJob).toHaveBeenCalledWith(job)
  })

  it("shows empty state when no jobs", () => {
    renderJobTable(
      <JobTable jobs={[]} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )
    expect(screen.getByText("No jobs yet")).toBeInTheDocument()
  })

  it("renders pagination when more than 10 jobs", () => {
    const jobs = Array.from({ length: 15 }, (_, i) =>
      makeJob({ id: `j${i}` }),
    )
    renderJobTable(
      <JobTable jobs={jobs} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument()
  })

  it("calls onPageChange when next page is clicked", () => {
    const onPageChange = vi.fn()
    const jobs = Array.from({ length: 15 }, (_, i) =>
      makeJob({ id: `j${i}` }),
    )
    renderJobTable(
      <JobTable jobs={jobs} page={0} onPageChange={onPageChange} onSelectJob={vi.fn()} />,
    )
    fireEvent.click(screen.getByLabelText("Next page"))
    expect(onPageChange).toHaveBeenCalledWith(1)
  })

  it("shows job name from metadata when available", () => {
    const job = makeJob({ id: "j1", metadata: { name: "My Training Run" } })
    renderJobTable(
      <JobTable jobs={[job]} page={0} onPageChange={vi.fn()} onSelectJob={vi.fn()} />,
    )
    expect(screen.getByText("My Training Run")).toBeInTheDocument()
  })
})
