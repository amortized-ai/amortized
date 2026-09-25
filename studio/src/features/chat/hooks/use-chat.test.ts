import { renderHook, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { useChatStore } from "@/stores/chat-store"
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

vi.mock("@/lib/api-client", () => ({
  sendOpenCodeMessage: vi.fn(async () => mockResponse),
  fetchSessionMessages: vi.fn(async () => []),
  fetchPendingMessages: vi.fn(async () => []),
  generateChatTitle: vi.fn(async (msg: string) => msg.slice(0, 40)),
  createJob: vi.fn(async () => ({ id: "job-1", status: "queued" })),
  getBaseUrl: vi.fn(() => ""),
}))

import { useChat } from "./use-chat"

function resetStore() {
  useChatStore.setState({
    currentConversationId: null,
    conversations: [],
    sessionMap: {},
    sessionStatus: {},
    notifiedJobs: {},
    jobInFlight: {},
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

    let sendPromise: Promise<void>
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
