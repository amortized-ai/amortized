import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
function Wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
import { ActionCard } from "./action-card"
import { MessageBubble } from "./message-bubble"
import type { ChatMessage } from "../types"

describe("ActionCard — training compute display (issue #442)", () => {
  const gpuConfig = {
    algorithm: "sft",
    model_name_or_path: "meta-llama/Llama-3.2-1B",
    num_train_epochs: 1,
    nproc_per_node: 1,
  }

  function trainingAction(config: Record<string, unknown>) {
    return {
      action: "Submit training job",
      description: "Train model on dataset",
      params: config,
      jobType: "training" as const,
      endpoint: "/api/v1/jobs/training",
      config,
    }
  }

  it("renders Compute: CPU row for CPU jobs and no GPUs row", () => {
    render(
      <ActionCard
        action={trainingAction({ ...gpuConfig, device: "cpu" })}
        onConfirm={vi.fn()}
        onReject={vi.fn()}
      />,
    )
    expect(screen.getByText("Compute")).toBeInTheDocument()
    expect(screen.getByText("CPU")).toBeInTheDocument()
    expect(screen.queryByText("GPUs")).not.toBeInTheDocument()
  })

  it("renders GPUs row unchanged for GPU jobs", () => {
    render(
      <ActionCard
        action={trainingAction(gpuConfig)}
        onConfirm={vi.fn()}
        onReject={vi.fn()}
      />,
    )
    expect(screen.getByText("GPUs")).toBeInTheDocument()
    expect(screen.getByText("GPUs").closest("tr")).toHaveTextContent("1")
    expect(screen.queryByText("Compute")).not.toBeInTheDocument()
  })

  it("renders and submits old persisted action cards without device as GPU", () => {
    // Backward compat with message-replay paths (issues #440/#414): configs
    // persisted before the device field existed must keep rendering as GPU jobs.
    const onConfirm = vi.fn()
    render(
      <ActionCard
        action={trainingAction(gpuConfig)}
        onConfirm={onConfirm}
        onReject={vi.fn()}
      />,
    )
    expect(screen.getByText("GPUs")).toBeInTheDocument()
    expect(screen.queryByText("Compute")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("Confirm"))
    expect(onConfirm).toHaveBeenCalledOnce()
  })
})

describe("MessageBubble — VRAM estimate vs CPU mode (issue #442)", () => {
  const vramToolResults = [{
    name: "show_vram_estimate",
    result: JSON.stringify({
      estimates: [
        { model_size: "1B", method: "sft", vram_per_gpu_gb: 16, vram_range: "12-20 GB" },
      ],
    }),
    collapsed: true,
  }]

  function makeMessage(config: Record<string, unknown> | undefined): ChatMessage {
    return {
      id: "1",
      role: "assistant",
      content: "Here's the estimate:",
      timestamp: new Date().toISOString(),
      toolResults: vramToolResults,
      proposedAction: config
        ? {
            action: "Create TRAINING Job",
            description: "Submit this training job?",
            params: config,
            jobType: "training",
            endpoint: "/api/v1/jobs/training",
            config,
          }
        : null,
      optionCards: [],
    }
  }

  it("shows the VRAM estimate card for GPU jobs", () => {
    render(
      <MessageBubble
        message={makeMessage({ algorithm: "sft", nproc_per_node: 1 })}
        onConfirmAction={vi.fn()}
        onRejectAction={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("VRAM Estimate")).toBeInTheDocument()
    expect(screen.queryByText(/CPU mode/)).not.toBeInTheDocument()
  })

  it("suppresses the VRAM estimate card for CPU jobs and shows the CPU note", () => {
    render(
      <MessageBubble
        message={makeMessage({ algorithm: "sft", device: "cpu", nproc_per_node: 1 })}
        onConfirmAction={vi.fn()}
        onRejectAction={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.queryByText("VRAM Estimate")).not.toBeInTheDocument()
    expect(
      screen.getByText("CPU mode — tiny models only · expect slow training"),
    ).toBeInTheDocument()
  })

  it("shows the VRAM estimate card for old messages without a proposed action", () => {
    render(
      <MessageBubble
        message={makeMessage(undefined)}
        onConfirmAction={vi.fn()}
        onRejectAction={vi.fn()}
      />,
      { wrapper: Wrapper },
    )
    expect(screen.getByText("VRAM Estimate")).toBeInTheDocument()
  })
})

