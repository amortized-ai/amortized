import { render, screen } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { GpuUtilizationCard } from "./gpu-utilization-card"

vi.mock("@/features/jobs/api/use-gpu-allocation", () => ({
  useGpuAllocation: vi.fn(),
}))

import { useGpuAllocation } from "@/features/jobs/api/use-gpu-allocation"

const mockUseGpuAllocation = vi.mocked(useGpuAllocation)

function renderCard() {
  return render(
    <MemoryRouter>
      <GpuUtilizationCard />
    </MemoryRouter>,
  )
}

describe("GpuUtilizationCard", () => {
  it("renders loading skeleton with aria-busy", () => {
    mockUseGpuAllocation.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      refetch: vi.fn(),
    } as ReturnType<typeof useGpuAllocation>)

    renderCard()

    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true")
    expect(screen.getByText("Loading GPU allocation data...")).toBeInTheDocument()
  })

  it("renders error state with retry button", () => {
    mockUseGpuAllocation.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      refetch: vi.fn(),
    } as ReturnType<typeof useGpuAllocation>)

    renderCard()

    expect(screen.getByText("Cannot reach the backend server")).toBeInTheDocument()
    expect(screen.getByText("Try again")).toBeInTheDocument()
  })

  it("renders no-GPU state when available is false", () => {
    mockUseGpuAllocation.mockReturnValue({
      data: {
        available: false,
        reason: "No GPU-capable compute backend configured",
        total_gpus: 0,
        allocated_gpus: 0,
        total_memory_requested_gib: 0,
        gpu_devices: [],
        jobs: [],
      },
      isLoading: false,
      isError: false,
      refetch: vi.fn(),
    } as ReturnType<typeof useGpuAllocation>)

    renderCard()

    expect(screen.getByText("No GPU detected on this server")).toBeInTheDocument()
    expect(screen.getByText("Check Settings")).toBeInTheDocument()
  })

  it("renders populated state with GPU allocation data", () => {
    mockUseGpuAllocation.mockReturnValue({
      data: {
        available: true,
        reason: null,
        total_gpus: 3,
        allocated_gpus: 3,
        total_memory_requested_gib: 176,
        gpu_devices: [],
        jobs: [
          {
            job_id: "j1",
            job_name: "sentiment",
            status: "running",
            gpus_requested: 2,
            gpu_type: "A100",
            memory_requested_gib: 160,
          },
          {
            job_id: "j2",
            job_name: "classify",
            status: "queued",
            gpus_requested: 1,
            gpu_type: "T4",
            memory_requested_gib: 16,
          },
        ],
      },
      isLoading: false,
      isError: false,
      refetch: vi.fn(),
    } as ReturnType<typeof useGpuAllocation>)

    renderCard()

    expect(screen.getByText("GPU Allocation")).toBeInTheDocument()
    expect(screen.getByText("176 GiB")).toBeInTheDocument()
    expect(screen.getByText("sentiment")).toBeInTheDocument()
    expect(screen.getByText("classify")).toBeInTheDocument()
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "3")
  })

  it("renders empty allocation state when available but no jobs", () => {
    mockUseGpuAllocation.mockReturnValue({
      data: {
        available: true,
        reason: null,
        total_gpus: 0,
        allocated_gpus: 0,
        total_memory_requested_gib: 0,
        gpu_devices: [],
        jobs: [],
      },
      isLoading: false,
      isError: false,
      refetch: vi.fn(),
    } as ReturnType<typeof useGpuAllocation>)

    renderCard()

    expect(screen.getByText("All GPUs are available")).toBeInTheDocument()
  })
})
