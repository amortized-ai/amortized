import { renderHook, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
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
