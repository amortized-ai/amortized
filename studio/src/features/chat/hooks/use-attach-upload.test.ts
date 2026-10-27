import { renderHook, waitFor, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createElement, type ReactNode } from "react"
import { toast } from "sonner"
import { useAttachUpload } from "./use-attach-upload"
import { useChatStore } from "@/stores/chat-store"

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
  // Status + pending references are stored keyed by the active conversation; give tests one.
  useChatStore.setState({ currentConversationId: "test-conv", attachStatuses: {}, pendingReferences: {} })
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

  it("tracks concurrent attaches and delivers both references, self-driven", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/datasets/upload")) return jsonResponse(makeJob({ id: "ds-job" }), 202)
      if (url.includes("/jobs/ds-job"))
        return jsonResponse(makeJob({ id: "ds-job", status: "succeeded", mlflow_run_id: "ds-run" }))
      if (url.includes("/jobs/doc-job"))
        return jsonResponse(makeJob({ id: "doc-job", status: "succeeded", mlflow_run_id: "doc-run" }))
      return jsonResponse({})
    })
    // The drain advances off each send's completion, so both references land without
    // the chat's canSend being toggled — serialized, neither dropped.
    const onReference = vi.fn()
    const { result } = renderHook(() => useAttachUpload(onReference), { wrapper: createWrapper() })

    act(() => {
      result.current.attachDataset(new File(["{}"], "d.jsonl"))
      result.current.attachDocument("doc-job", "s.pdf")
    })

    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(2))
    const msgs = onReference.mock.calls.map((c) => c[0] as string).join("\n")
    expect(msgs).toContain("run_id: ds-run")
    expect(msgs).toContain("document_id: doc-run")
  })

  it("re-queues a reference when the send reports it was not delivered", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/jobs/doc-job"))
        return jsonResponse(makeJob({ id: "doc-job", status: "succeeded", mlflow_run_id: "doc-run" }))
      return jsonResponse({})
    })
    // First dispatch reports not-delivered (false) -> the reference must stay queued.
    const onReference = vi.fn().mockReturnValueOnce(false).mockReturnValue(true)
    const { result, rerender } = renderHook(
      ({ canSend }: { canSend: boolean }) => useAttachUpload(onReference, canSend),
      { wrapper: createWrapper(), initialProps: { canSend: true } },
    )

    act(() => {
      result.current.attachDocument("doc-job", "spec.pdf")
    })

    // Dispatched once and rejected -> the same reference is still queued, not consumed.
    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(1))
    // An idle transition resumes the drain; the retry is accepted.
    act(() => rerender({ canSend: false }))
    act(() => rerender({ canSend: true }))
    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(2))
    expect(onReference.mock.calls[1]![0]).toBe(onReference.mock.calls[0]![0])
    expect(onReference.mock.calls[1]![0] as string).toContain("document_id: doc-run")
  })

  it("queues a reference while the chat is busy and flushes it when idle", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/jobs/doc-job"))
        return jsonResponse(makeJob({ id: "doc-job", status: "succeeded", mlflow_run_id: "doc-run" }))
      return jsonResponse({})
    })
    const onReference = vi.fn()
    const { result, rerender } = renderHook(
      ({ canSend }: { canSend: boolean }) => useAttachUpload(onReference, canSend),
      { wrapper: createWrapper(), initialProps: { canSend: false } },
    )

    act(() => {
      result.current.attachDocument("doc-job", "spec.pdf")
    })

    // The job resolves (upload finishes), but the chat is busy -> queued, not sent.
    await waitFor(() => expect(result.current.isUploading).toBe(false))
    expect(onReference).not.toHaveBeenCalled()

    // Chat goes idle -> the queued reference flushes.
    act(() => {
      rerender({ canSend: true })
    })
    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(1))
    expect(onReference.mock.calls[0]![0] as string).toContain("document_id: doc-run")
  })

  it("keeps a conversation's status when switching away and back (survives the remount)", async () => {
    // getJob never resolves, so the upload stays in-flight and the status persists.
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/jobs/doc-job")) return new Promise<Response>(() => {})
      return jsonResponse({})
    })
    useChatStore.setState({ currentConversationId: "conv-A", attachStatuses: {} })
    const onReference = vi.fn()
    const { result } = renderHook(() => useAttachUpload(onReference), { wrapper: createWrapper() })

    act(() => {
      result.current.attachDocument("doc-job", "spec.pdf")
    })
    await waitFor(() =>
      expect(result.current.statuses.map((s) => s.message)).toContain("Processing spec.pdf"),
    )

    // Switch to another conversation -> this conversation shows no status.
    act(() => useChatStore.setState({ currentConversationId: "conv-B" }))
    await waitFor(() => expect(result.current.isUploading).toBe(false))

    // Switch back -> the original conversation's status is still there.
    act(() => useChatStore.setState({ currentConversationId: "conv-A" }))
    await waitFor(() =>
      expect(result.current.statuses.map((s) => s.message)).toContain("Processing spec.pdf"),
    )
  })

  it("delivers a queued reference after switching conversations away and back", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/jobs/doc-job"))
        return jsonResponse(makeJob({ id: "doc-job", status: "succeeded", mlflow_run_id: "doc-run" }))
      return jsonResponse({})
    })
    useChatStore.setState({ currentConversationId: "conv-A", attachStatuses: {}, pendingReferences: {} })
    const onReference = vi.fn().mockReturnValue(true)
    const { result, rerender } = renderHook(
      ({ canSend }: { canSend: boolean }) => useAttachUpload(onReference, canSend),
      { wrapper: createWrapper(), initialProps: { canSend: false } },
    )

    // Attach in conv-A while the chat is busy -> the reference is queued, not delivered.
    act(() => {
      result.current.attachDocument("doc-job", "spec.pdf")
    })
    await waitFor(() => expect(useChatStore.getState().pendingReferences["conv-A"]?.length).toBe(1))
    expect(onReference).not.toHaveBeenCalled()

    // Switch to another conversation -> the queued reference survives (still keyed to A).
    act(() => useChatStore.setState({ currentConversationId: "conv-B" }))
    expect(useChatStore.getState().pendingReferences["conv-A"]?.length).toBe(1)

    // Switch back to A and go idle -> delivered, and only to A.
    act(() => {
      useChatStore.setState({ currentConversationId: "conv-A" })
      rerender({ canSend: true })
    })
    await waitFor(() => expect(onReference).toHaveBeenCalledTimes(1))
    expect(onReference.mock.calls[0]![0] as string).toContain("document_id: doc-run")
    expect(useChatStore.getState().pendingReferences["conv-A"]).toBeUndefined()
  })
})
