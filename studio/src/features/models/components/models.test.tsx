import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ModelTable } from "./model-table"
import type { ModelRecord } from "@/types/api"

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
      <ModelTable
        models={models}
        page={0}
        onPageChange={vi.fn()}
        onSelectModel={vi.fn()}
      />,
      { wrapper },
    )
    expect(screen.getByText("model-a")).toBeInTheDocument()
    expect(screen.getByText("model-b")).toBeInTheDocument()
  })

  it("shows empty state when no models", () => {
    render(
      <ModelTable
        models={[]}
        page={0}
        onPageChange={vi.fn()}
        onSelectModel={vi.fn()}
      />,
      { wrapper },
    )
    expect(screen.getByText("No models yet")).toBeInTheDocument()
  })

  it("calls onSelectModel when row clicked", () => {
    const onSelect = vi.fn()
    const model = makeModel({ name: "click-me" })
    render(
      <ModelTable
        models={[model]}
        page={0}
        onPageChange={vi.fn()}
        onSelectModel={onSelect}
      />,
      { wrapper },
    )
    fireEvent.click(screen.getByTestId("model-row-click-me"))
    expect(onSelect).toHaveBeenCalledWith(model)
  })

  it("renders pagination when more than 10 models", () => {
    const models = Array.from({ length: 15 }, (_, i) =>
      makeModel({ name: `model-${i}` }),
    )
    render(
      <ModelTable
        models={models}
        page={0}
        onPageChange={vi.fn()}
        onSelectModel={vi.fn()}
      />,
      { wrapper },
    )
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument()
  })
})
