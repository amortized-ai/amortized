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
  currentConversationId: null,
  setCurrentConversationId: mockSetCurrentConversationId,
  conversations: [],
  addConversation: mockAddConversation,
  deleteConversation: mockDeleteConversation,
  updateConversationTitle: mockUpdateConversationTitle,
  addMessage: mockAddMessage,
  updateMessage: mockUpdateMessage,
  getConversationMessages: mockGetConversationMessages,
  _hasHydrated: true,
}

const mockSetSessionStatus = vi.fn()

vi.mock("@/features/settings", () => ({
  useGatewayRoutes: () => ({ data: [], isLoading: false }),
}))

vi.mock("@/stores/chat-store", () => ({
  useChatStore: Object.assign(
    () => mockStoreValue,
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
    expect(assistantMsg.toolResults[0]!.name).toBe("get jobs")
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
