import { renderHook, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { useChatStore } from "@/stores/chat-store"
import { useSettingsStore } from "@/stores/settings-store"
import { encodeModelSelection } from "../models"
import type { OpenCodeResponse } from "../types"

// Only the network layer is mocked — the real zustand store is used, so these tests
// exercise the actual store<->hook integration (message ownership, the send lock, and
// navigation losslessness). Mocking the store would hide exactly the bugs this covers.
let mockResponse: OpenCodeResponse = {
  info: {
    providerID: "mock",
    modelID: "mock-model",
    cost: 0,
    tokens: { input: 10, output: 20, reasoning: 0 },
    finish: "stop",
    id: "msg_mock",
    sessionID: "ses_mock",
  },
  parts: [],
}

vi.mock("@/lib/api-client", async (importOriginal) => {
  // Keep the real module (notably the ApiError class, used for instanceof checks in
  // use-chat) and override only the network functions.
  const actual = await importOriginal<typeof import("@/lib/api-client")>()
  return {
    ...actual,
    sendOpenCodeMessage: vi.fn(async () => mockResponse),
    fetchSessionMessages: vi.fn(async () => []),
    fetchPendingMessages: vi.fn(async () => []),
    generateChatTitle: vi.fn(async (msg: string) => msg.slice(0, 40)),
    createJob: vi.fn(async () => ({ id: "job-1", status: "queued" })),
    getJob: vi.fn(async (id: string) => ({ id, status: "running" })),
    getBaseUrl: vi.fn(() => ""),
  }
})

import { useChat } from "./use-chat"

function resetStore() {
  useChatStore.setState({
    currentConversationId: null,
    conversations: [],
    sessionMap: {},
    sessionStatus: {},
    notifiedJobs: {},
    jobInFlight: {},
    runningJob: {},
    activeRequests: {},
    requestErrors: {},
    drafts: {},
    _hasHydrated: true,
  })
}

type SeedMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
  toolResults?: { name: string; result: string; collapsed: boolean }[]
}
function seedConversation(id: string, messages: SeedMessage[]) {
  useChatStore.setState({
    conversations: [{ id, title: "Test", created_at: "2026-01-01", updated_at: "2026-01-01", messages }],
    currentConversationId: id,
  })
}

beforeEach(() => {
  mockResponse = {
    info: {
      providerID: "mock",
      modelID: "mock-model",
      cost: 0,
      tokens: { input: 10, output: 20, reasoning: 0 },
      finish: "stop",
      id: "msg_mock",
      sessionID: "ses_mock",
    },
    parts: [],
  }
  resetStore()
  vi.clearAllMocks()
})

describe("useChat", () => {
  it("starts in idle state with no messages", () => {
    const { result } = renderHook(() => useChat())
    expect(result.current.chatState).toBe("idle")
    expect(result.current.messages).toEqual([])
    expect(result.current.isStreaming).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it("sends a user message and receives an assistant response (via the real store)", async () => {
    mockResponse.parts = [{ type: "text", text: "Hello! I can help you." }]

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Hello")
    })

    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[0]!.role).toBe("user")
    expect(result.current.messages[0]!.content).toBe("Hello")
    expect(result.current.messages[1]!.role).toBe("assistant")
    expect(result.current.messages[1]!.content).toBe("Hello! I can help you.")

    // The messages live in the store — not a private local copy.
    const convId = useChatStore.getState().currentConversationId!
    expect(useChatStore.getState().getConversationMessages(convId)).toHaveLength(2)
  })

  it("parses tool invocations in the response", async () => {
    mockResponse.parts = [
      {
        type: "text",
        text: 'Let me check.\n\n<function_calls>\n<invoke name="mcp_amortized__get_jobs">\n</invoke>\n</function_calls>\n<function_response>\n[{"id":"job-1","status":"running"}]\n</function_response>\n\nFound 1 running job.',
      },
    ]

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Show running jobs")
    })

    const assistantMsg = result.current.messages[1]!
    expect(assistantMsg.content).toBe("Let me check.\n\nFound 1 running job.")
    expect(assistantMsg.toolResults).toHaveLength(1)
    expect(assistantMsg.toolResults[0]!.name).toBe("get_jobs")
  })

  // Regression for the send-lock leak: the lock was acquired under the generated convId but
  // released under lockId ("" for a new conversation), so the second message in every new
  // conversation was permanently blocked.
  it("allows a SECOND message in a newly created conversation", async () => {
    mockResponse.parts = [{ type: "text", text: "reply" }]

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("First")
    })
    expect(result.current.messages).toHaveLength(2)

    await act(async () => {
      await result.current.sendMessage("Second")
    })
    // Must not be blocked by a leaked lock: two full turns.
    expect(result.current.messages).toHaveLength(4)
    expect(result.current.messages[2]!.role).toBe("user")
    expect(result.current.messages[2]!.content).toBe("Second")
  })

  // Regression: a new conversation's first send has no prior history, so it must settle to
  // "connected", not "rebuilt" (which would show a spurious context-rebuilt warning).
  it("does not mark a new conversation's first send as rebuilt", async () => {
    mockResponse.parts = [{ type: "text", text: "hi" }]

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("hello")
    })

    const convId = useChatStore.getState().currentConversationId!
    expect(useChatStore.getState().getSessionStatus(convId)).toBe("connected")
  })

  // The PR's purpose (Issue #401): a mid-response unmount/remount must not lose messages.
  it("preserves messages across unmount/remount (store is the source of truth)", async () => {
    mockResponse.parts = [{ type: "text", text: "kept" }]

    const first = renderHook(() => useChat())
    await act(async () => {
      await first.result.current.sendMessage("hello")
    })
    expect(first.result.current.messages).toHaveLength(2)
    first.unmount()

    // A fresh hook (the remounted component) derives the same messages from the store.
    const second = renderHook(() => useChat())
    expect(second.result.current.messages).toHaveLength(2)
    expect(second.result.current.messages[1]!.content).toBe("kept")
  })

  // Regression for finding #7: the failed turn must survive, not vanish.
  it("keeps the assistant turn with the error persisted on failure", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("Session expired"))

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Fail")
    })

    expect(result.current.error).toBe("Session expired")
    expect(result.current.chatState).toBe("error")
    // The assistant placeholder is retained with the error written into it (persisted).
    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[1]!.role).toBe("assistant")
    expect(result.current.messages[1]!.content).toContain("Session expired")
  })

  it("does not restore an error banner from a persisted context-replay error flag", () => {
    seedConversation("restored-error", [
      { id: "u1", role: "user", content: "hi", timestamp: "2026-01-01" },
      { id: "a1", role: "assistant", content: "Error: Session expired", timestamp: "2026-01-01" },
    ])
    useChatStore.getState().updateMessageFields("restored-error", "a1", { error: true })

    const { result } = renderHook(() => useChat())
    expect(result.current.messages[1]!.content).toBe("Error: Session expired")
    expect(result.current.chatState).toBe("done")
    expect(result.current.error).toBeNull()
    expect(result.current.isStreaming).toBe(false)
  })

  // Regression for findings #2/#3: an orphaned empty placeholder (a send lost to a full
  // reload) is marked interrupted on mount — no infinite poll, no spurious recovery.
  it("marks an orphaned empty assistant placeholder as interrupted on mount", () => {
    seedConversation("conv-orphan", [
      { id: "u1", role: "user", content: "hi", timestamp: "2026-01-01" },
      { id: "a1", role: "assistant", content: "", timestamp: "2026-01-01" },
    ])

    const { result } = renderHook(() => useChat())

    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[1]!.content).toBe(
      "Response was interrupted. Send a new message to continue.",
    )
    // Not stuck streaming.
    expect(result.current.isStreaming).toBe(false)
  })

  // A completed tool-only turn (empty text but tool results) must NOT be treated as orphaned.
  it("leaves a completed tool-only turn untouched on mount", () => {
    seedConversation("conv-tool", [
      { id: "u1", role: "user", content: "options?", timestamp: "2026-01-01" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        timestamp: "2026-01-01",
        toolResults: [{ name: "present_options", result: "{}", collapsed: true }],
      },
    ])

    const { result } = renderHook(() => useChat())
    expect(result.current.messages[1]!.content).toBe("")
    expect(result.current.messages[1]!.toolResults).toHaveLength(1)
  })

  it("does not send while already streaming", async () => {
    let resolveMessage: (() => void) | null = null
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockImplementationOnce(
      () =>
        new Promise<OpenCodeResponse>((resolve) => {
          resolveMessage = () => resolve({ ...mockResponse, parts: [{ type: "text", text: "Done" }] })
        }),
    )

    const { result } = renderHook(() => useChat())

    let sendPromise: Promise<boolean>
    act(() => {
      sendPromise = result.current.sendMessage("First")
    })
    await act(async () => {
      await result.current.sendMessage("Second while streaming")
    })

    // The second send is rejected while the first is in flight: just the first turn.
    expect(result.current.messages).toHaveLength(2)

    await act(async () => {
      resolveMessage?.()
      await sendPromise!
    })
  })

  it.each(["success", "failure"])("keeps a returning conversation locked until its pending response settles (%s)", async (outcome) => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    let resolve!: (response: OpenCodeResponse) => void
    let reject!: (error: Error) => void
    vi.mocked(sendOpenCodeMessage).mockImplementationOnce(() => new Promise((res, rej) => {
      resolve = res
      reject = rej
    }))
    seedConversation("thinking", [])
    useChatStore.getState().addConversation({
      id: "other", title: "Other", created_at: "2026-01-01", updated_at: "2026-01-01", messages: [],
    })
    const first = renderHook(() => useChat())
    let pending!: Promise<boolean>
    act(() => { pending = first.result.current.sendMessage("First") })
    first.unmount()

    act(() => { useChatStore.getState().setCurrentConversationId("other") })
    const other = renderHook(() => useChat())
    expect(other.result.current.isStreaming).toBe(false)
    await act(async () => { expect(await other.result.current.sendMessage("Other chat")).toBe(true) })
    other.unmount()

    act(() => { useChatStore.getState().setCurrentConversationId("thinking") })
    const returned = renderHook(() => useChat())
    expect(returned.result.current.isStreaming).toBe(true)
    expect(returned.result.current.thinkingStep).toBeTruthy()
    await act(async () => { expect(await returned.result.current.sendMessage("Too soon")).toBe(false) })
    expect(returned.result.current.messages).toHaveLength(2)
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(2)

    await act(async () => {
      if (outcome === "failure") reject(new Error("Turn failed"))
      else resolve({ ...mockResponse, parts: [{ type: "text", text: "Finished" }] })
      await pending
    })
    // Completion must update the newly mounted view without a reconciliation timer.
    expect(returned.result.current.isStreaming).toBe(false)
    expect(returned.result.current.thinkingStep).toBeNull()
    expect(returned.result.current.chatState).toBe(outcome === "failure" ? "error" : "done")
    expect(returned.result.current.messages[1]!.content).toContain(outcome === "failure" ? "Turn failed" : "Finished")
    await act(async () => { expect(await returned.result.current.sendMessage("Next")).toBe(true) })
    expect(returned.result.current.messages).toHaveLength(4)
  })

  it("shares the thinking state with another mounted view of the same conversation", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    let resolve!: (response: OpenCodeResponse) => void
    vi.mocked(sendOpenCodeMessage).mockImplementationOnce(() => new Promise((res) => { resolve = res }))
    seedConversation("shared", [])
    const sender = renderHook(() => useChat())
    const observer = renderHook(() => useChat())
    let pending!: Promise<boolean>
    act(() => { pending = sender.result.current.sendMessage("First") })
    try {
      expect(observer.result.current.isStreaming).toBe(true)
      await act(async () => { expect(await observer.result.current.sendMessage("Too soon")).toBe(false) })
      expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    } finally {
      await act(async () => {
        resolve({ ...mockResponse, parts: [{ type: "text", text: "Finished" }] })
        await pending
      })
    }
    expect(observer.result.current.isStreaming).toBe(false)
  })

  it("keeps a new conversation locked through its first keyed remount and session fetch", async () => {
    const { fetchSessionMessages, sendOpenCodeMessage } = await import("@/lib/api-client")
    let resolve!: (response: OpenCodeResponse[]) => void
    vi.mocked(fetchSessionMessages).mockImplementationOnce(() => new Promise((res) => { resolve = res }))
    mockResponse.parts = [{ type: "text", text: "Finished" }]
    const first = renderHook(() => useChat())
    let pending!: Promise<boolean>
    await act(async () => { pending = first.result.current.sendMessage("First") })
    first.unmount()

    const returned = renderHook(() => useChat())
    expect(returned.result.current.isStreaming).toBe(true)
    await act(async () => { expect(await returned.result.current.sendMessage("Too soon")).toBe(false) })
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    await act(async () => {
      resolve([])
      await pending
    })
    expect(returned.result.current.isStreaming).toBe(false)
    expect(returned.result.current.messages[1]!.content).toBe("Finished")
  })

  it("keeps a job-completion response locked across remounts", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    let resolve!: (response: OpenCodeResponse) => void
    vi.mocked(sendOpenCodeMessage).mockImplementationOnce(() => new Promise((res) => { resolve = res }))
    seedConversation("job-complete", [])
    useChatStore.getState().setJobInFlight("job-complete", true)
    const first = renderHook(() => useChat())
    let pending!: Promise<void>
    act(() => { pending = first.result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    first.unmount()

    const returned = renderHook(() => useChat())
    expect(returned.result.current.jobInFlight).toBe(false)
    expect(returned.result.current.isStreaming).toBe(true)
    await act(async () => { expect(await returned.result.current.sendMessage("Too soon")).toBe(false) })
    await act(async () => {
      resolve({ ...mockResponse, parts: [{ type: "text", text: "Evaluation complete" }] })
      await pending
    })
    expect(returned.result.current.isStreaming).toBe(false)
    expect(returned.result.current.messages).toHaveLength(1)
    expect(returned.result.current.messages[0]!.content).toBe("Evaluation complete")
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
  })

  it("clears a failed submission's error when cancelling after switching away and back", async () => {
    const { createJob } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(new Error("Submission failed"))
    seedConversation("confirm", [{ id: "action", role: "assistant", content: "Confirm?", timestamp: "2026-01-01" }])
    useChatStore.getState().updateMessageFields("confirm", "action", {
      proposedAction: { action: "create_eval_job", description: "Confirm?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" },
    })
    useChatStore.getState().addConversation({
      id: "other", title: "Other", created_at: "2026-01-01", updated_at: "2026-01-01", messages: [],
    })
    useChatStore.getState().setRequestError("other", "Unrelated error")
    const first = renderHook(() => useChat())
    await act(async () => { await first.result.current.confirmAction() })
    expect(first.result.current.chatState).toBe("error")
    first.unmount()

    act(() => { useChatStore.getState().setCurrentConversationId("other") })
    const other = renderHook(() => useChat())
    other.unmount()
    act(() => { useChatStore.getState().setCurrentConversationId("confirm") })
    const returned = renderHook(() => useChat())
    expect(returned.result.current.error).toBe("Submission failed")
    await act(async () => { await returned.result.current.rejectAction() })

    expect(returned.result.current.proposedAction).toBeNull()
    expect(returned.result.current.error).toBeNull()
    expect(returned.result.current.chatState).toBe("done")
    expect(returned.result.current.isStreaming).toBe(false)
    expect(useChatStore.getState().requestErrors).toEqual({ other: "Unrelated error" })
  })

  it("prevents duplicate submissions and sends after remounting during job confirmation", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    let reject!: (error: Error) => void
    vi.mocked(createJob).mockImplementationOnce(() => new Promise((_resolve, rej) => { reject = rej }))
    seedConversation("confirm", [{ id: "action", role: "assistant", content: "Confirm?", timestamp: "2026-01-01" }])
    useChatStore.getState().updateMessageFields("confirm", "action", {
      proposedAction: { action: "create_eval_job", description: "Confirm?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" },
    })
    const first = renderHook(() => useChat())
    let pending!: Promise<void>
    act(() => { pending = first.result.current.confirmAction() })
    first.unmount()

    const returned = renderHook(() => useChat())
    expect(returned.result.current.isStreaming).toBe(true)
    await act(async () => {
      expect(await returned.result.current.sendMessage("Too soon")).toBe(false)
      await returned.result.current.confirmAction()
    })
    expect(createJob).toHaveBeenCalledTimes(1)
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
    await act(async () => {
      reject(new Error("Submission failed"))
      await pending
    })
    expect(returned.result.current.isStreaming).toBe(false)
    expect(returned.result.current.error).toBe("Submission failed")
    await act(async () => { expect(await returned.result.current.sendMessage("Try something else")).toBe(true) })
    expect(returned.result.current.error).toBeNull()
  })
})

describe("useChat — storage failure recovery", () => {
  afterEach(() => { vi.restoreAllMocks() })

  function failStorage() {
    const error = new DOMException("Storage quota exceeded", "QuotaExceededError")
    const write = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw error })
    return { error, write }
  }

  it("rolls back a failed send acquisition while preserving existing locks and errors", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("storage", [])
    useChatStore.getState().tryStartRequest("other")
    useChatStore.getState().setRequestError("other", "Other error")
    useChatStore.getState().setRequestError("storage", "Earlier error")
    const { result } = renderHook(() => useChat())
    const { error, write } = failStorage()

    expect(useChatStore.getState().tryStartRequest("other")).toBe(false)
    expect(write).not.toHaveBeenCalled()
    await act(async () => { await expect(result.current.sendMessage("First")).rejects.toBe(error) })
    expect(useChatStore.getState().activeRequests).toEqual({ other: true })
    expect(useChatStore.getState().requestErrors).toEqual({ other: "Other error", storage: "Earlier error" })
    expect(result.current.isStreaming).toBe(false)
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()

    write.mockRestore()
    await act(async () => { expect(await result.current.sendMessage("Retry")).toBe(true) })
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    expect(useChatStore.getState().activeRequests).toEqual({ other: true })
  })

  it("allows a remounted chat to send after a real storage quota overflow is removed", async () => {
    seedConversation("storage", [])
    const first = renderHook(() => useChat())
    act(() => {
      expect(() => useChatStore.getState().setDraft("storage", "x".repeat(6_000_000))).toThrow()
    })
    await act(async () => {
      await expect(first.result.current.sendMessage("First")).rejects.toMatchObject({ name: "QuotaExceededError" })
    })
    // The overflowing draft is in memory even though persistence failed.
    act(() => { useChatStore.getState().setDraft("storage", "") })
    expect(useChatStore.getState().activeRequests).toEqual({})
    first.unmount()

    const returned = renderHook(() => useChat())
    expect(returned.result.current.isStreaming).toBe(false)
    await act(async () => { expect(await returned.result.current.sendMessage("Retry")).toBe(true) })
  })

  it("releases a failed confirmation acquisition so submission can be retried", async () => {
    const { createJob } = await import("@/lib/api-client")
    seedConversation("storage", [{ id: "a1", role: "assistant", content: "Confirm?", timestamp: "2026-01-01" }])
    useChatStore.getState().updateMessageFields("storage", "a1", {
      proposedAction: { action: "create_eval_job", description: "Confirm?", params: {}, endpoint: "/api/v1/jobs/eval", jobType: "eval" },
    })
    const { result } = renderHook(() => useChat())
    const { error, write } = failStorage()
    await act(async () => { await expect(result.current.confirmAction()).rejects.toBe(error) })
    expect(result.current.isStreaming).toBe(false)
    expect(createJob).not.toHaveBeenCalled()

    write.mockRestore()
    await act(async () => { await result.current.confirmAction() })
    expect(createJob).toHaveBeenCalledTimes(1)
    expect(useChatStore.getState().activeRequests).toEqual({})
  })

  it("resets the notification queue after acquisition fails", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("storage", [])
    const { result } = renderHook(() => useChat())
    const { error, write } = failStorage()
    await act(async () => {
      await expect(result.current.notifyJobComplete("job-1", "eval", "succeeded")).rejects.toBe(error)
    })
    expect(useChatStore.getState().activeRequests).toEqual({})
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()

    write.mockRestore()
    await act(async () => { await result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    expect(useChatStore.getState().getNotifiedJobs("storage")).toEqual(["job-1"])
  })

  it.each([2, 3, 4, 5])("releases notification bookkeeping when setup storage write %s fails", async (failureWrite) => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("storage", [])
    const { result } = renderHook(() => useChat())
    const error = new DOMException("Storage quota exceeded", "QuotaExceededError")
    const originalWrite = Storage.prototype.setItem
    let writes = 0
    const write = vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key, value) {
      // Acquisition succeeds; fail job status, placeholder insertion, or model pinning.
      // Keep failing through catch/finally to exercise unconditional in-memory cleanup.
      if (++writes >= failureWrite) throw error
      originalWrite.call(this, key, value)
    })
    await act(async () => {
      await expect(result.current.notifyJobComplete("job-1", "eval", "succeeded")).rejects.toBe(error)
    })
    expect(useChatStore.getState().activeRequests).toEqual({})
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()

    write.mockRestore()
    await act(async () => { await result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    expect(useChatStore.getState().getNotifiedJobs("storage")).toEqual(["job-1"])
  })
})

describe("useChat — blocked job notifications", () => {
  it.each(["action", "request"])("retains a notification blocked by a pending %s for longer than a minute", async (blocker) => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.useFakeTimers()
    try {
      seedConversation("blocked-job", [
        { id: "a1", role: "assistant", content: "Review the next step", timestamp: "2026-01-01" },
      ])
      if (blocker === "action") {
        useChatStore.getState().updateMessageFields("blocked-job", "a1", {
          proposedAction: { action: "create_eval_job", description: "Confirm?", params: {}, endpoint: "/api/v1/jobs/eval" },
        })
      } else {
        useChatStore.getState().tryStartRequest("blocked-job")
      }
      useChatStore.getState().setJobInFlight("blocked-job", true)
      const first = renderHook(() => useChat())
      let pending!: Promise<void>
      act(() => { pending = first.result.current.notifyJobComplete("job-1", "eval", "succeeded") })
      first.unmount()
      const returned = renderHook(() => useChat())

      // Focus can clear the job flag while the notification is still blocked.
      act(() => { window.dispatchEvent(new Event("focus")) })
      expect(returned.result.current.jobInFlight).toBe(false)
      await act(async () => { await vi.advanceTimersByTimeAsync(61_000) })
      expect(sendOpenCodeMessage).not.toHaveBeenCalled()

      await act(async () => {
        if (blocker === "action") await returned.result.current.rejectAction()
        else useChatStore.getState().finishRequest("blocked-job")
        await vi.advanceTimersByTimeAsync(500)
        await pending
      })
      expect(sendOpenCodeMessage).toHaveBeenCalledExactlyOnceWith(
        "blocked-job", expect.any(String), expect.any(String), "job_complete", "succeeded", "eval", undefined,
      )
      expect(useChatStore.getState().getNotifiedJobs("blocked-job")).toEqual(["job-1"])
      returned.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it("delivers a queued job only once when a remounted monitor reports it again", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("blocked-job", [])
    useChatStore.getState().tryStartRequest("blocked-job")
    const first = renderHook(() => useChat())
    let firstPending!: Promise<void>
    act(() => { firstPending = first.result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    first.unmount()

    const returned = renderHook(() => useChat())
    let secondPending!: Promise<void>
    act(() => { secondPending = returned.result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
    await act(async () => {
      useChatStore.getState().finishRequest("blocked-job")
      await Promise.all([firstPending, secondPending])
    })

    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    expect(useChatStore.getState().getNotifiedJobs("blocked-job")).toEqual(["job-1"])
  })

  it("stops waiting without sending when the queued conversation is deleted", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("deleted-job", [])
    useChatStore.getState().tryStartRequest("deleted-job")
    const first = renderHook(() => useChat())
    let pending!: Promise<void>
    act(() => { pending = first.result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    first.unmount()

    await act(async () => {
      useChatStore.getState().deleteConversation("deleted-job")
      await pending
    })
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
    expect(useChatStore.getState().activeRequests).toEqual({})
  })
})

describe("useChat — split_dataset monitor flow", () => {
  it("keeps split_dataset tool results from session messages", async () => {
    const { fetchSessionMessages } = await import("@/lib/api-client")
    mockResponse.parts = [{ type: "text", text: "The split is running." }]
    ;(fetchSessionMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { info: { role: "user" }, parts: [] },
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "mcp_amortized__split_dataset",
            output: JSON.stringify({
              id: "9e2c1f70-1111-4222-8333-444455556666",
              type: "upload",
              status: "queued",
              config: {},
            }),
          },
        ],
      },
    ])

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Split off 20% for eval")
    })

    const assistantMsg = result.current.messages[1]!
    const splitTool = assistantMsg.toolResults.find((t) => t.name === "split_dataset")
    expect(splitTool).toBeDefined()
    expect(JSON.parse(splitTool!.result).id).toBe("9e2c1f70-1111-4222-8333-444455556666")
  })
})

describe("useChat — validate retry keeps the confirmation card", () => {
  it("builds proposedAction from the successful retry, not the errored first call", async () => {
    const { fetchSessionMessages } = await import("@/lib/api-client")
    mockResponse.parts = [{ type: "text", text: "The config is valid — see the card." }]
    ;(fetchSessionMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { info: { role: "user" }, parts: [] },
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "amortized_validate_eval_job",
            output: JSON.stringify({ code: "http_422", message: "judge endpoint is required to score the rubric" }),
          },
        ],
      },
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "amortized_validate_eval_job",
            output: JSON.stringify({
              valid: true,
              job_type: "eval",
              config: { eval_data_run_id: "a".repeat(32), rubric: [{ name: "accuracy", description: "d" }] },
              parent_job_id: "",
              recipe: "",
              warnings: [],
            }),
          },
        ],
      },
    ])

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Run the eval")
    })

    const assistantMsg = result.current.messages[1]!
    const validates = assistantMsg.toolResults.filter((t) => t.name === "validate_eval_job")
    expect(validates).toHaveLength(2)
    expect(assistantMsg.proposedAction).not.toBeNull()
    expect(assistantMsg.proposedAction!.jobType).toBe("eval")
    expect(assistantMsg.proposedAction!.endpoint).toBe("/api/v1/jobs/eval")
    expect(assistantMsg.proposedAction!.config!.eval_data_run_id).toBe("a".repeat(32))
  })
})

describe("useChat — job-finished notify keeps the confirmation card", () => {
  it("builds proposedAction when a failed job self-heals and re-validates in the notify turn", async () => {
    const { fetchSessionMessages } = await import("@/lib/api-client")
    mockResponse.parts = [{ type: "text", text: "Fixed the config — confirm the card to re-run." }]
    ;(fetchSessionMessages as ReturnType<typeof vi.fn>).mockResolvedValue([
      { info: { role: "user" }, parts: [{ type: "text", text: "Job job-xyz (SDG) finished with status: failed." }] },
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "amortized_validate_sdg_job",
            output: "Input validation error: max_tokens not accepted by gpt-5",
          },
        ],
      },
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "amortized_validate_sdg_job",
            output: JSON.stringify({
              valid: true,
              job_type: "sdg",
              config: { num_samples: 10 },
              parent_job_id: "",
              recipe: "",
              warnings: [],
            }),
          },
        ],
      },
    ])

    seedConversation("conv-notify", [
      { id: "u1", role: "user", content: "run sdg", timestamp: "2026-01-01" },
      { id: "a1", role: "assistant", content: "ok", timestamp: "2026-01-01" },
    ])

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.notifyJobComplete("job-xyz", "sdg", "failed")
    })

    const last = result.current.messages[result.current.messages.length - 1]!
    expect(last.role).toBe("assistant")
    expect(last.proposedAction).not.toBeNull()
    expect(last.proposedAction!.jobType).toBe("sdg")
    expect(last.proposedAction!.endpoint).toBe("/api/v1/jobs/sdg")
  })
})

describe("useChat — per-conversation model", () => {
  it("pins a new chat to the global default model on first send", async () => {
    const MODEL = encodeModelSelection("openai", "gpt-5.6-sol")
    useSettingsStore.setState({ chatModelSelection: MODEL })
    const { sendOpenCodeMessage } = await import("@/lib/api-client")

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Hi")
    })

    // Sent with the global default, and both the conversation and the assistant
    // message are stamped from that send-time value.
    expect(sendOpenCodeMessage).toHaveBeenCalledWith(expect.any(String), "Hi", MODEL)
    const convId = useChatStore.getState().currentConversationId!
    const conv = useChatStore.getState().conversations.find((c) => c.id === convId)!
    expect(conv.model).toBe(MODEL)
    expect(conv.messages.find((m) => m.role === "assistant")!.model).toBe(MODEL)
  })

  it("uses the conversation's pinned model over the global default", async () => {
    const PINNED = encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default")
    const GLOBAL = encodeModelSelection("openai", "gpt-5.6-sol")
    useSettingsStore.setState({ chatModelSelection: GLOBAL })
    useChatStore.setState({
      conversations: [
        { id: "c1", title: "T", created_at: "2026-01-01", updated_at: "2026-01-01", messages: [], model: PINNED },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage } = await import("@/lib/api-client")

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("Hi")
    })

    expect(sendOpenCodeMessage).toHaveBeenCalledWith("c1", "Hi", PINNED)
    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    expect(conv.model).toBe(PINNED)
    expect(conv.messages.find((m) => m.role === "assistant")!.model).toBe(PINNED)
  })

  it("applies the conversation model to job-completion turns", async () => {
    mockResponse.parts = [{ type: "text", text: "Here are your next steps." }]
    const PINNED = encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default")
    useChatStore.setState({
      conversations: [
        {
          id: "c1",
          title: "T",
          created_at: "2026-01-01",
          updated_at: "2026-01-01",
          model: PINNED,
          messages: [
            { id: "u1", role: "user", content: "hi", timestamp: "2026-01-01" },
            { id: "a1", role: "assistant", content: "ok", timestamp: "2026-01-01" },
          ],
        },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage } = await import("@/lib/api-client")

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.notifyJobComplete("job-1", "sdg", "succeeded")
    })

    // The job-completion send carries the conversation's model, and its new assistant
    // turn is stamped with it (not left unlabeled or defaulted to the global).
    expect(sendOpenCodeMessage).toHaveBeenCalledWith(
      "c1",
      expect.any(String),
      PINNED,
      "job_complete",
      "succeeded",
      "sdg",
      undefined,
    )
    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    const lastAssistant = [...conv.messages].reverse().find((m) => m.role === "assistant")!
    expect(lastAssistant.model).toBe(PINNED)
  })

  it("names the unavailable model when a job-completion turn fails on it", async () => {
    const PINNED = encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default")
    useChatStore.setState({
      conversations: [
        {
          id: "c1",
          title: "T",
          created_at: "2026-01-01",
          updated_at: "2026-01-01",
          model: PINNED,
          messages: [{ id: "u1", role: "user", content: "hi", timestamp: "2026-01-01" }],
        },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage, ApiError } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new ApiError(400, "Agent turn failed", "That model isn't available on this provider/endpoint.", "model_unavailable"),
    )

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.notifyJobComplete("job-1", "sdg", "succeeded")
    })

    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    const last = conv.messages[conv.messages.length - 1]!
    expect(last.content).toContain("Claude Sonnet 5")
    expect(last.content).toMatch(/could not be reached/i)
    expect(last.error).toBe(true) // excluded from context replay
    expect(result.current.chatState).toBe("done")
    expect(result.current.error).toBeNull()
  })

  it("shows a model-named message when a user send fails on an unavailable model", async () => {
    const PINNED = encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default")
    useChatStore.setState({
      conversations: [
        { id: "c1", title: "T", created_at: "2026-01-01", updated_at: "2026-01-01", model: PINNED, messages: [] },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage, ApiError } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new ApiError(400, "Agent turn failed", "That model isn't available on this provider/endpoint.", "model_unavailable"),
    )

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("hi")
    })

    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    const assistant = conv.messages.find((m) => m.role === "assistant")!
    expect(assistant.error).toBe(true)
    expect(assistant.content).toContain("Claude Sonnet 5")
    expect(assistant.content).toMatch(/could not be reached/i)
    expect(assistant.content).not.toContain("Error:")
  })

  it("abstracts an auth failure without leaking key/auth details", async () => {
    const PINNED = encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default")
    useChatStore.setState({
      conversations: [
        { id: "c1", title: "T", created_at: "2026-01-01", updated_at: "2026-01-01", model: PINNED, messages: [] },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage, ApiError } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new ApiError(401, "Agent turn failed", "That model isn't available with your API key or project (auth/permission).", "auth"),
    )

    const { result } = renderHook(() => useChat())
    await act(async () => {
      await result.current.sendMessage("hi")
    })

    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    const assistant = conv.messages.find((m) => m.role === "assistant")!
    expect(assistant.content).toMatch(/could not be reached/i)
    // Security: never surface auth/key specifics to the chat user.
    expect(assistant.content).not.toMatch(/api key|auth|permission/i)
  })

  it("uses the generic fallback message for a non-model job-completion failure", async () => {
    useChatStore.setState({
      conversations: [
        {
          id: "c1",
          title: "T",
          created_at: "2026-01-01",
          updated_at: "2026-01-01",
          model: encodeModelSelection("google-vertex-anthropic", "claude-sonnet-5@default"),
          messages: [{ id: "u1", role: "user", content: "hi", timestamp: "2026-01-01" }],
        },
      ],
      currentConversationId: "c1",
    })
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("network timeout"))

    const { result, unmount } = renderHook(() => useChat())
    await act(async () => {
      await result.current.notifyJobComplete("job-1", "sdg", "succeeded")
    })

    const conv = useChatStore.getState().conversations.find((c) => c.id === "c1")!
    const last = conv.messages[conv.messages.length - 1]!
    expect(last.content).toBe("I wasn't able to suggest next steps. You can ask me what to do next.")
    expect(last.error).toBe(true) // excluded from context replay
    expect(result.current.chatState).toBe("done")
    expect(result.current.error).toBeNull()
    unmount()

    const returned = renderHook(() => useChat())
    expect(returned.result.current.chatState).toBe("done")
    expect(returned.result.current.error).toBeNull()
  })
})

describe("useChat — blocked confirmation (model check)", () => {
  const SANITIZED =
    "judge model 'gpt-oss' at 'http://gw/v1' did not answer a test request (unreachable: refused)"

  async function modelCheckError() {
    const { ApiError } = await import("@/lib/api-client")
    return new ApiError(
      422,
      "Unprocessable Entity",
      { code: "model_check_failed", message: SANITIZED, details: [{ field: "judge", error: "gpt-oss at http://gw/v1: unreachable" }] },
      "model_check_failed",
    )
  }

  function seedCard(convId = "confirm") {
    seedConversation(convId, [{ id: "action", role: "assistant", content: "Confirm?", timestamp: "2026-01-01" }])
    useChatStore.getState().updateMessageFields(convId, "action", {
      proposedAction: { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" },
    })
  }

  const actionMessage = (convId = "confirm") =>
    useChatStore.getState().getConversationMessages(convId).find((m) => m.id === "action")!

  afterEach(async () => {
    const { createJob, sendOpenCodeMessage, fetchSessionMessages } = await import("@/lib/api-client")
    vi.mocked(createJob).mockReset().mockImplementation(async () => ({ id: "job-1", status: "queued" }) as never)
    vi.mocked(sendOpenCodeMessage).mockReset().mockImplementation(async () => mockResponse)
    vi.mocked(fetchSessionMessages).mockReset().mockImplementation(async () => [])
  })

  it("invalidates the card and tells Morty once, with the sanitized failure and no banner", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    expect(actionMessage().proposedAction).toBeNull()
    expect(actionMessage().actionBlocked?.message).toBe(SANITIZED) // envelope message, not the detail bullets
    expect(result.current.error).toBeNull()
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    const [convId, text, , event, , jobType] = vi.mocked(sendOpenCodeMessage).mock.calls[0]!
    expect(convId).toBe("confirm")
    expect(event).toBe("confirmation_blocked")
    expect(jobType).toBe("eval")
    expect(text).toBe(`Eval submission was rejected before a job was created. ${SANITIZED}`)
  })

  it("renders a new confirmation card when Morty re-validates in the event turn", async () => {
    const { createJob, sendOpenCodeMessage, fetchSessionMessages } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    // The event turn's own messages (the server's result), never the session's latest.
    vi.mocked(sendOpenCodeMessage).mockResolvedValueOnce({
      ...mockResponse,
      parts: [],
      event_messages: [
        {
          info: { role: "assistant" },
          parts: [{
            type: "tool",
            tool: "amortized_validate_eval_job",
            output: JSON.stringify({ valid: true, job_type: "eval", config: { judge: { model: "gpt-5" } } }),
          }],
        },
      ],
    } as never)
    vi.mocked(fetchSessionMessages).mockResolvedValue([
      { info: { role: "user" }, parts: [{ type: "text", text: "later turn" }] },
    ] as never)
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    const last = result.current.messages.at(-1)!
    expect(last.proposedAction?.jobType).toBe("eval")
    expect(last.proposedAction?.endpoint).toBe("/api/v1/jobs/eval")
  })

  it("a Retry the server already ran renders nothing from later turns and counts as delivered", async () => {
    const { createJob, sendOpenCodeMessage, fetchSessionMessages, EventAlreadyDeliveredError } =
      await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(new Error("Cannot reach Morty"))
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("failed"))
    const before = useChatStore.getState().getConversationMessages("confirm").length

    // The first send did reach the server; later turns have since added a validate card to the
    // session. The replay must not resurrect it.
    vi.mocked(fetchSessionMessages).mockResolvedValue([
      { info: { role: "user" }, parts: [{ type: "text", text: "later turn" }] },
      {
        info: { role: "assistant" },
        parts: [{
          type: "tool",
          tool: "amortized_validate_eval_job",
          output: JSON.stringify({ valid: true, job_type: "eval", config: {} }),
        }],
      },
    ] as never)
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(
      new EventAlreadyDeliveredError("confirmation_blocked:action"),
    )
    act(() => { result.current.retryBlockedNotice("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    const messages = useChatStore.getState().getConversationMessages("confirm")
    expect(messages).toHaveLength(before) // the placeholder was removed, no reply rendered
    expect(messages.some((m) => m.proposedAction)).toBe(false)
    expect(fetchSessionMessages).not.toHaveBeenCalled()
  })

  it("a replayed event renders that turn's own reply, including its corrected card", async () => {
    const { createJob, sendOpenCodeMessage, fetchSessionMessages, EventAlreadyDeliveredError } =
      await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    // The first send got through but its response was lost; the resend comes back as a replay.
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(
      new EventAlreadyDeliveredError("confirmation_blocked:action", [
        {
          info: { role: "assistant" },
          parts: [
            { type: "text", text: "Switched the judge to a working model." },
            {
              type: "tool",
              tool: "amortized_validate_eval_job",
              output: JSON.stringify({ valid: true, job_type: "eval", config: { judge: { model: "gpt-5" } } }),
            },
          ],
        },
      ] as never),
    )
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    const last = result.current.messages.at(-1)!
    expect(last.content).toBe("Switched the judge to a working model.")
    expect(last.proposedAction?.jobType).toBe("eval")
    expect(fetchSessionMessages).not.toHaveBeenCalled() // never the session's latest messages
  })

  it("a replay after the conversation moved on shows the reply without a card", async () => {
    const { createJob, sendOpenCodeMessage, EventAlreadyDeliveredError } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(new Error("Cannot reach Morty"))
    seedCard()
    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("failed"))

    // The user carries on before retrying.
    useChatStore.getState().addMessage("confirm", { id: "later", role: "user", content: "never mind", timestamp: "2026-01-02" })
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(
      new EventAlreadyDeliveredError("confirmation_blocked:action", [
        {
          info: { role: "assistant" },
          parts: [
            { type: "text", text: "Here's a corrected config." },
            {
              type: "tool",
              tool: "amortized_validate_eval_job",
              output: JSON.stringify({ valid: true, job_type: "eval", config: {} }),
            },
          ],
        },
      ] as never),
    )
    act(() => { result.current.retryBlockedNotice("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    const last = useChatStore.getState().getConversationMessages("confirm").at(-1)!
    expect(last.content).toBe("Here's a corrected config.")
    expect(last.proposedAction).toBeNull()
  })

  it("Confirm submits the card that was clicked, not the newest one", async () => {
    const { createJob } = await import("@/lib/api-client")
    seedConversation("confirm", [
      { id: "older", role: "assistant", content: "Older card", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Newer card", timestamp: "2026-01-02" },
    ])
    const card = (model: string) => ({
      action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval",
      config: { judge: { model } }, jobType: "eval" as const,
    })
    useChatStore.getState().updateMessageFields("confirm", "older", { proposedAction: card("older-judge") })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card("newer-judge") })
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("older") })

    expect(createJob).toHaveBeenCalledWith("/api/v1/jobs/eval", { judge: { model: "older-judge" } })
    const msgs = useChatStore.getState().getConversationMessages("confirm")
    expect(msgs.find((m) => m.id === "older")!.proposedAction).toBeNull()
    expect(msgs.find((m) => m.id === "newer")!.proposedAction).not.toBeNull()
  })

  it("defers an older card's notice while a newer card is pending, then sends it", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    seedConversation("confirm", [
      { id: "action", role: "assistant", content: "Older card", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Newer card", timestamp: "2026-01-02" },
    ])
    const card = { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" as const }
    useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: card })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card })
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("deferred"))
    expect(sendOpenCodeMessage).not.toHaveBeenCalled() // no recovery turn beside the pending card

    await act(async () => { await result.current.rejectAction() }) // the newer card is dismissed
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
  })

  it("re-checks under the request lock: a card that appeared while waiting defers the notice", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    vi.mocked(sendOpenCodeMessage).mockRejectedValueOnce(new Error("Cannot reach Morty"))
    seedCard()
    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("failed"))
    vi.mocked(sendOpenCodeMessage).mockClear()

    // Another request holds the lock; while the Retry waits for it, a new card arrives.
    expect(useChatStore.getState().tryStartRequest("confirm")).toBe(true)
    act(() => { result.current.retryBlockedNotice("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("sending"))
    act(() => {
      useChatStore.getState().addMessage("confirm", { id: "newer", role: "assistant", content: "New card", timestamp: "2026-01-03" })
      useChatStore.getState().updateMessageFields("confirm", "newer", {
        proposedAction: { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" },
      })
      useChatStore.getState().finishRequest("confirm")
    })

    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("deferred"))
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
  })

  it("a successful Confirm keeps older notices deferred until the job completes", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    seedConversation("confirm", [
      { id: "action", role: "assistant", content: "Older card", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Newer card", timestamp: "2026-01-02" },
    ])
    const card = { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" as const }
    useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: card })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card })
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") }) // rejected
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("deferred"))
    await act(async () => { await result.current.confirmAction("newer") }) // created: a job runs

    expect(useChatStore.getState().jobInFlight.confirm).toBe(true)
    expect(actionMessage().actionBlocked?.notify).toBe("deferred") // no mid-job wakeup
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()

    await act(async () => { await result.current.notifyJobComplete("job-1", "eval", "succeeded") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))
    const events = vi.mocked(sendOpenCodeMessage).mock.calls.map((call) => call[3])
    expect(events).toEqual(["job_complete", "confirmation_blocked"]) // the job's wakeup first
  })

  it("keeps a schema rejection's field details for the card and Morty", async () => {
    const { ApiError, createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(
      new ApiError(422, "Unprocessable Entity", {
        code: "validation_error",
        message: "Request validation failed",
        details: [{ loc: ["body", "rubric", 0, "name"], msg: "Field required", type: "missing" }],
      }, "validation_error"),
    )
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    const expected = "Request validation failed\n- rubric.0.name: Field required"
    expect(actionMessage().actionBlocked?.message).toBe(expected)
    expect(vi.mocked(sendOpenCodeMessage).mock.calls[0]![1]).toContain("- rubric.0.name: Field required")
  })

  it("a finished job releases the running-job flag even while Morty's notification waits", async () => {
    seedCard()
    useChatStore.getState().setJobInFlight("confirm", true)
    expect(useChatStore.getState().tryStartRequest("confirm")).toBe(true) // the lock is busy
    const { result } = renderHook(() => useChat())

    // Not awaited: the notification waits for the lock (and the pending card) to clear.
    act(() => { void result.current.notifyJobComplete("job-9", "eval", "succeeded") })

    expect(useChatStore.getState().jobInFlight.confirm ?? false).toBe(false)
    act(() => {
      useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: null })
      useChatStore.getState().finishRequest("confirm")
    })
    await vi.waitFor(() => expect(useChatStore.getState().getNotifiedJobs("confirm")).toContain("job-9"))
  })

  it("an older job's completion doesn't unlock or wake Morty during a newer job", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("confirm", [
      { id: "action", role: "assistant", content: "Card A", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Card B", timestamp: "2026-01-02" },
    ])
    const card = { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" as const }
    useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: card })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card })
    vi.mocked(createJob).mockResolvedValueOnce({ id: "job-A", status: "queued" } as never)
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") }) // job A runs
    // A finishes; its notification waits behind the still-pending card B.
    act(() => { void result.current.notifyJobComplete("job-A", "eval", "succeeded") })
    vi.mocked(createJob).mockResolvedValueOnce({ id: "job-B", status: "queued" } as never)
    await act(async () => { await result.current.confirmAction("newer") }) // job B runs

    await act(async () => { await new Promise((r) => setTimeout(r, 20)) })
    expect(useChatStore.getState().jobInFlight.confirm).toBe(true) // still B's
    expect(sendOpenCodeMessage).not.toHaveBeenCalled() // A's wakeup waits for B

    await act(async () => { await result.current.notifyJobComplete("job-B", "eval", "succeeded") })
    await vi.waitFor(() => expect(useChatStore.getState().getNotifiedJobs("confirm")).toEqual(
      expect.arrayContaining(["job-A", "job-B"]),
    ))
    expect(useChatStore.getState().jobInFlight.confirm ?? false).toBe(false)
  })

  it("refocusing unlocks the composer but keeps the running job until it is verified terminal", async () => {
    const { createJob, getJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    seedConversation("confirm", [
      { id: "action", role: "assistant", content: "Card A", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Card B", timestamp: "2026-01-02" },
    ])
    const card = { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" as const }
    useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: card })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card })
    vi.mocked(createJob).mockResolvedValueOnce({ id: "job-A", status: "queued" } as never)
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })
    act(() => { void result.current.notifyJobComplete("job-A", "eval", "succeeded") })
    vi.mocked(createJob).mockResolvedValueOnce({ id: "job-B", status: "queued" } as never)
    await act(async () => { await result.current.confirmAction("newer") })

    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await new Promise((r) => setTimeout(r, 20))
    })
    expect(getJob).toHaveBeenCalledWith("job-B")
    expect(result.current.jobInFlight).toBe(false) // the composer is usable again
    expect(useChatStore.getState().runningJob.confirm).toBe("job-B")
    expect(sendOpenCodeMessage).not.toHaveBeenCalled() // A's wakeup still waits for B

    await act(async () => { await result.current.notifyJobComplete("job-B", "eval", "succeeded") })
    await vi.waitFor(() => expect(useChatStore.getState().getNotifiedJobs("confirm")).toEqual(
      expect.arrayContaining(["job-A", "job-B"]),
    ))
  })

  it.each(["finished", "gone"])("refocusing releases a running job that is %s", async (state) => {
    const { getJob, ApiError } = await import("@/lib/api-client")
    if (state === "finished") vi.mocked(getJob).mockResolvedValueOnce({ id: "job-9", status: "succeeded" } as never)
    else vi.mocked(getJob).mockRejectedValueOnce(new ApiError(404, "Not Found", null))
    seedConversation("confirm", [])
    useChatStore.getState().setRunningJob("confirm", "job-9")
    useChatStore.getState().setJobInFlight("confirm", true)
    renderHook(() => useChat())

    act(() => { window.dispatchEvent(new Event("focus")) })

    await vi.waitFor(() => expect(useChatStore.getState().runningJob.confirm).toBeUndefined())
    expect(useChatStore.getState().jobInFlight.confirm ?? false).toBe(false)
  })

  it("refocusing keeps the running job when its status can't be checked", async () => {
    const { getJob } = await import("@/lib/api-client")
    vi.mocked(getJob).mockRejectedValueOnce(new TypeError("Failed to fetch"))
    seedConversation("confirm", [])
    useChatStore.getState().setRunningJob("confirm", "job-9")
    useChatStore.getState().setJobInFlight("confirm", true)
    renderHook(() => useChat())

    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await new Promise((r) => setTimeout(r, 20))
    })

    expect(useChatStore.getState().runningJob.confirm).toBe("job-9")
    expect(useChatStore.getState().jobInFlight.confirm ?? false).toBe(false)
  })

  it("Confirm is still refused after refocus while the job is running", async () => {
    const { createJob } = await import("@/lib/api-client")
    seedCard()
    useChatStore.getState().setRunningJob("confirm", "job-9")
    useChatStore.getState().setJobInFlight("confirm", true)
    const { result } = renderHook(() => useChat())

    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await new Promise((r) => setTimeout(r, 20))
    })
    expect(result.current.jobInFlight).toBe(false)
    await act(async () => { await result.current.confirmAction("action") })

    expect(createJob).not.toHaveBeenCalled()
    expect(actionMessage().proposedAction).not.toBeNull()
    expect(result.current.error).toMatch(/already running/)
  })

  it("won't start a second job while one is running in the conversation", async () => {
    const { createJob } = await import("@/lib/api-client")
    seedCard()
    useChatStore.getState().setJobInFlight("confirm", true)
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })

    expect(createJob).not.toHaveBeenCalled()
    expect(actionMessage().proposedAction).not.toBeNull() // the card stays
    expect(useChatStore.getState().jobInFlight.confirm).toBe(true) // the running job's flag
    expect(result.current.error).toMatch(/already running/)
  })

  it("an ordinary reply that dismisses the pending card resumes the deferred notice", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    seedConversation("confirm", [
      { id: "action", role: "assistant", content: "Older card", timestamp: "2026-01-01" },
      { id: "newer", role: "assistant", content: "Newer card", timestamp: "2026-01-02" },
    ])
    const card = { action: "Create EVAL Job", description: "Submit?", params: {}, endpoint: "/api/v1/jobs/eval", config: {}, jobType: "eval" as const }
    useChatStore.getState().updateMessageFields("confirm", "action", { proposedAction: card })
    useChatStore.getState().updateMessageFields("confirm", "newer", { proposedAction: card })
    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("deferred"))

    // The user types instead of confirming the newer card; sending dismisses it.
    await act(async () => { await result.current.sendMessage("never mind, use another judge") })

    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))
    const events = vi.mocked(sendOpenCodeMessage).mock.calls.map((call) => call[3])
    expect(events.at(-1)).toBe("confirmation_blocked")
  })

  it("an event send never renders the session's latest messages", async () => {
    const { createJob, fetchSessionMessages } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    vi.mocked(fetchSessionMessages).mockResolvedValue([
      { info: { role: "user" }, parts: [{ type: "text", text: "another turn" }] },
      {
        info: { role: "assistant" },
        parts: [{ type: "tool", tool: "amortized_validate_eval_job", output: JSON.stringify({ valid: true, job_type: "eval", config: {} }) }],
      },
    ] as never)
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    expect(fetchSessionMessages).not.toHaveBeenCalled()
    const msgs = useChatStore.getState().getConversationMessages("confirm")
    expect(msgs.some((m) => m.proposedAction)).toBe(false)
  })

  it("makes one automatic delivery, then offers Retry that resends the same event", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    vi.mocked(sendOpenCodeMessage).mockRejectedValue(new Error("Cannot reach Morty"))
    seedCard()
    const { result } = renderHook(() => useChat())
    const before = useChatStore.getState().getConversationMessages("confirm").length

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("failed"))

    // sendOpenCodeMessage owns transport retries (idempotent via the event id); the hook doesn't
    // add another layer on top.
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1)
    expect(actionMessage().actionBlocked?.message).toBe(SANITIZED)
    // a failed delivery leaves no error turn behind
    expect(useChatStore.getState().getConversationMessages("confirm")).toHaveLength(before)

    vi.mocked(sendOpenCodeMessage).mockImplementation(async () => mockResponse)
    act(() => { result.current.retryBlockedNotice("action") })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))
    expect(sendOpenCodeMessage).toHaveBeenCalledTimes(2)
    const eventIds = vi.mocked(sendOpenCodeMessage).mock.calls.map((call) => call[7]?.eventId)
    expect(eventIds).toEqual(["confirmation_blocked:action", "confirmation_blocked:action"])
  })

  it("delivers to the conversation the card was confirmed in, even after switching away", async () => {
    const { createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(await modelCheckError())
    let release!: () => void
    vi.mocked(sendOpenCodeMessage).mockImplementationOnce(
      () => new Promise((resolve) => { release = () => resolve(mockResponse) }),
    )
    seedCard()
    useChatStore.getState().addConversation({
      id: "other", title: "Other", created_at: "2026-01-01", updated_at: "2026-01-01", messages: [],
    })
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(sendOpenCodeMessage).toHaveBeenCalledTimes(1))
    act(() => { useChatStore.getState().setCurrentConversationId("other") })
    await act(async () => { release() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    expect(vi.mocked(sendOpenCodeMessage).mock.calls[0]![0]).toBe("confirm")
    expect(useChatStore.getState().getConversationMessages("other")).toHaveLength(0)
    expect(useChatStore.getState().getConversationMessages("confirm").at(-1)!.role).toBe("assistant")
  })

  it("does not resend a notice interrupted by a reload; it offers Retry instead", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    seedCard()
    useChatStore.getState().updateMessageFields("confirm", "action", {
      proposedAction: null,
      actionBlocked: { message: SANITIZED, jobType: "eval", notify: "sending" },
    })

    renderHook(() => useChat())

    expect(actionMessage().actionBlocked?.notify).toBe("failed")
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
  })

  it("any other rejection of the create (a 4xx) is handled the same way, with the server's reason", async () => {
    // e.g. an upstream job that failed or was deleted after validation — only possible at Confirm.
    const { ApiError, createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    const reason = "parent_job_id: job 'sdg-1' has status 'failed' (must be 'succeeded')"
    vi.mocked(createJob).mockRejectedValueOnce(
      new ApiError(422, "Unprocessable Entity", { code: "http_422", message: reason, details: [] }, "http_422"),
    )
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })
    await vi.waitFor(() => expect(actionMessage().actionBlocked?.notify).toBe("delivered"))

    expect(result.current.error).toBeNull()
    expect(actionMessage().proposedAction).toBeNull()
    expect(actionMessage().actionBlocked?.message).toBe(reason)
    const [, text, , event] = vi.mocked(sendOpenCodeMessage).mock.calls[0]!
    expect(event).toBe("confirmation_blocked")
    expect(text).toBe(`Eval submission was rejected before a job was created. ${reason}`)
  })

  it.each([
    ["a server error", 500],
    ["an auth failure", 401],
    ["rate limiting", 429],
  ])("keeps the banner for %s: not about the job, and the card stays", async (_label, status) => {
    const { ApiError, createJob, sendOpenCodeMessage } = await import("@/lib/api-client")
    vi.mocked(createJob).mockRejectedValueOnce(
      new ApiError(status, "Error", { code: `http_${status}`, message: "try again" }, `http_${status}`),
    )
    seedCard()
    const { result } = renderHook(() => useChat())

    await act(async () => { await result.current.confirmAction() })

    expect(result.current.error).toBe("try again")
    expect(actionMessage().proposedAction).not.toBeNull()
    expect(actionMessage().actionBlocked).toBeUndefined()
    expect(sendOpenCodeMessage).not.toHaveBeenCalled()
  })
})

describe("useChat — session warmup keeps a session through transient failures", () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  function seedWithSession() {
    seedConversation("warm", [{ id: "m1", role: "user", content: "hi", timestamp: "2026-01-01" }])
    useChatStore.getState().setSessionId("warm", "ses-1")
  }

  it.each([
    ["a gateway error", async () => new Response("", { status: 502 })],
    ["a network failure", async () => { throw new TypeError("fetch failed") }],
  ])("keeps the session on %s (it also holds the server's delivered-event record)", async (_label, health) => {
    const fetchMock = vi.fn(health)
    globalThis.fetch = fetchMock as never
    seedWithSession()
    renderHook(() => useChat())

    await vi.waitFor(() => expect(useChatStore.getState().sessionStatus.warm).toBe("unknown"))
    expect(useChatStore.getState().getSessionId("warm")).toBe("ses-1")
    expect(fetchMock).toHaveBeenCalledTimes(1) // no replacement session was created
  })

  it("rebuilds only when the server no longer knows the session (404)", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "POST" && url.endsWith("/agent/session")) return new Response(JSON.stringify({ id: "ses-2" }))
      if (init?.method === "POST") return new Response("{}")
      return new Response("", { status: 404 })
    })
    globalThis.fetch = fetchMock as never
    seedWithSession()
    renderHook(() => useChat())

    await vi.waitFor(() => expect(useChatStore.getState().getSessionId("warm")).toBe("ses-2"))
  })
})
