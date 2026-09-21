import { renderHook, act } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import type { OpenCodeResponse } from "../types"

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
  generateChatTitle: vi.fn(async (msg: string) => msg.slice(0, 40)),
}))

const mockGetConversationMessages = vi.fn(() => [])
const mockSetCurrentConversationId = vi.fn()
const mockAddConversation = vi.fn()
const mockDeleteConversation = vi.fn()
const mockUpdateConversationTitle = vi.fn()
const mockAddMessage = vi.fn()
const mockUpdateMessage = vi.fn()

const mockStoreValue = {
  currentConversationId: null as string | null,
  setCurrentConversationId: mockSetCurrentConversationId,
  conversations: [],
  addConversation: mockAddConversation,
  deleteConversation: mockDeleteConversation,
  updateConversationTitle: mockUpdateConversationTitle,
  addMessage: mockAddMessage,
  updateMessage: mockUpdateMessage,
  getConversationMessages: mockGetConversationMessages,
  jobInFlight: {},
  _hasHydrated: true,
}

const mockSetSessionStatus = vi.fn()

vi.mock("@/stores/chat-store", () => ({
  useChatStore: Object.assign(
    (selector?: (s: typeof mockStoreValue) => unknown) =>
      selector ? selector(mockStoreValue) : mockStoreValue,
    {
      getState: () => ({
        currentConversationId: null,
        conversations: [],
        getSessionId: vi.fn(() => undefined),
        setSessionId: vi.fn(),
        clearSessionId: vi.fn(),
        setSessionStatus: mockSetSessionStatus,
        getSessionStatus: vi.fn(() => "unknown"),
        getConversationMessages: mockGetConversationMessages,
        updateMessageFields: vi.fn(),
        removeMessage: vi.fn(),
        setJobInFlight: vi.fn(),
        addNotifiedJob: vi.fn(),
        getNotifiedJobs: vi.fn(() => [] as string[]),
      }),
    },
  ),
}))

import { useChat } from "./use-chat"

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

  it("sends a user message and receives assistant response", async () => {
    mockResponse.parts = [
      { type: "text", text: "Hello! I can help you." },
    ]

    const { result } = renderHook(() => useChat())

    await act(async () => {
      await result.current.sendMessage("Hello")
    })

    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[0]!.role).toBe("user")
    expect(result.current.messages[0]!.content).toBe("Hello")
    expect(result.current.messages[1]!.role).toBe("assistant")
    expect(result.current.messages[1]!.content).toBe("Hello! I can help you.")
  })

  it("handles tool invocations in response parts", async () => {
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

  it("handles API errors", async () => {
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("Session expired"),
    )

    const { result } = renderHook(() => useChat())

    await act(async () => {
      await result.current.sendMessage("Fail")
    })

    expect(result.current.error).toBe("Session expired")
    expect(result.current.chatState).toBe("error")
  })

  it("does not send while already streaming", async () => {
    let resolveMessage: (() => void) | null = null
    const { sendOpenCodeMessage } = await import("@/lib/api-client")
    ;(sendOpenCodeMessage as ReturnType<typeof vi.fn>).mockImplementationOnce(
      () =>
        new Promise<OpenCodeResponse>((resolve) => {
          resolveMessage = () =>
            resolve({
              ...mockResponse,
              parts: [{ type: "text", text: "Done" }],
            })
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

    expect(result.current.messages).toHaveLength(2)

    await act(async () => {
      resolveMessage?.()
      await sendPromise!
    })
  })
})

describe("useChat — split_dataset monitor flow", () => {
  it("keeps split_dataset tool results from session messages", async () => {
    // Turn results only carry step markers; the split_dataset tool part
    // arrives via fetchSessionMessages — it must survive into the
    // message's toolResults so the monitor card can render.
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
    // jobInFlight blocks the "done" state while the split runs
    expect(mockStoreValue.jobInFlight).toBeDefined()
  })
})

describe("useChat — validate retry keeps the confirmation card", () => {
  it("builds proposedAction from the successful retry, not the errored first call", async () => {
    // Regression: the agent's first validate_eval_job errored (422), it
    // fixed the config and re-validated in the SAME turn. The dedup used
    // to keep only the errored result, so no confirmation card rendered
    // and the agent started fabricating cards with present_options.
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
            output: JSON.stringify({
              code: "http_422",
              message: "judge endpoint is required to score the rubric",
            }),
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
    // Both results survive (retries are not deduped away)
    expect(validates).toHaveLength(2)
    // The confirmation card is built from the successful one
    expect(assistantMsg.proposedAction).not.toBeNull()
    expect(assistantMsg.proposedAction!.jobType).toBe("eval")
    expect(assistantMsg.proposedAction!.endpoint).toBe("/api/v1/jobs/eval")
    expect(assistantMsg.proposedAction!.config!.eval_data_run_id).toBe("a".repeat(32))
  })
})

describe("useChat — job-finished notify keeps the confirmation card", () => {
  it("builds proposedAction when a failed job self-heals and re-validates in the notify turn", async () => {
    // Regression: when a job finishes, the client auto-notifies the agent
    // ("Job ... finished with status: ...") and the agent may fix the config and
    // re-validate in that turn. That turn is handled by processJobNotifyQueue,
    // which used to never build proposedAction — so the user saw the agent say
    // "confirm the card" with no card, and had to ask it to resubmit.
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

    mockStoreValue.currentConversationId = "conv-notify"
    try {
      const { result } = renderHook(() => useChat())

      await act(async () => {
        await result.current.notifyJobComplete("job-xyz", "sdg", "failed")
      })

      const last = result.current.messages[result.current.messages.length - 1]!
      expect(last.role).toBe("assistant")
      // The confirmation card is built from the successful re-validation
      expect(last.proposedAction).not.toBeNull()
      expect(last.proposedAction!.jobType).toBe("sdg")
      expect(last.proposedAction!.endpoint).toBe("/api/v1/jobs/sdg")
    } finally {
      mockStoreValue.currentConversationId = null
    }
  })
})
