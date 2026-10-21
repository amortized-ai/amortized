import { renderHook, waitFor, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createElement, type ReactNode } from "react"
import { toast } from "sonner"
import { useAttachUpload } from "./use-attach-upload"

vi.mock("sonner", () => ({
  toast: {
    loading: vi.fn(() => "toast-id"),
    success: vi.fn(),
    error: vi.fn(),
  },
}))

const mockFetch = vi.fn()
const originalFetch = globalThis.fetch

beforeEach(() => {
  globalThis.fetch = mockFetch
  mockFetch.mockReset()
  vi.mocked(toast.loading).mockClear()
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
})

afterEach(() => {
  globalThis.fetch = originalFetch
})

function jsonResponse(data: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  )
}

function makeJob(overrides: Record<string, unknown>) {
  return {
    id: "job-1",
    type: "upload",
    status: "queued",
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
    created_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  }
}

function createWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

describe("useAttachUpload", () => {
  it("uploads a dataset, waits for the job, and posts a run_id reference", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/datasets/upload")) return jsonResponse(makeJob({ id: "ds-job" }), 202)
      if (url.includes("/jobs/"))
        return jsonResponse(makeJob({ id: "ds-job", status: "succeeded", mlflow_run_id: "run-abc" }))
      return jsonResponse({})
    })
    const onReference = vi.fn()
    const { result } = renderHook(() => useAttachUpload(onReference), { wrapper: createWrapper() })

    act(() => {
      result.current.attachDataset(new File(["{}"], "intents.jsonl"))
    })

    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(1))
    const msg = onReference.mock.calls[0]![0] as string
    expect(msg).toContain("dataset")
    expect(msg).toContain("intents.jsonl")
    expect(msg).toContain("run_id: run-abc")
  })

  it("does not post a reference when the dataset job fails", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/datasets/upload")) return jsonResponse(makeJob({ id: "ds-job" }), 202)
      if (url.includes("/jobs/"))
        return jsonResponse(makeJob({ id: "ds-job", status: "failed", error: "bad file" }))
      return jsonResponse({})
    })
    const onReference = vi.fn()
    const { result } = renderHook(() => useAttachUpload(onReference), { wrapper: createWrapper() })

    act(() => {
      result.current.attachDataset(new File(["{}"], "bad.jsonl"))
    })

    await waitFor(() => expect(vi.mocked(toast.error)).toHaveBeenCalled())
    expect(onReference).not.toHaveBeenCalled()
  })

  it("tracks an already-submitted document job and posts a document_id reference", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/jobs/"))
        return jsonResponse(makeJob({ id: "doc-job", status: "succeeded", mlflow_run_id: "doc-run" }))
      return jsonResponse({})
    })
    const onReference = vi.fn()
    const { result } = renderHook(() => useAttachUpload(onReference), { wrapper: createWrapper() })

    act(() => {
      result.current.attachDocument("doc-job", "spec.pdf")
    })

    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(1))
    const msg = onReference.mock.calls[0]![0] as string
    expect(msg).toContain("document")
    expect(msg).toContain("spec.pdf")
    expect(msg).toContain("document_id: doc-run")
  })
})
