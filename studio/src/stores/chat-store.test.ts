import { describe, it, expect, beforeEach } from "vitest"

const STORAGE_KEY = "amortized-chat"

function createMockStorage(): Storage {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
    get length() {
      return Object.keys(store).length
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  }
}

const mockStorage = createMockStorage()

Object.defineProperty(window, "localStorage", {
  value: mockStorage,
  writable: true,
})

// import AFTER localStorage is defined so zustand persist binds to it
const { useChatStore } = await import("./chat-store")
import type { PersistedConversation } from "./chat-store"

function storage() {
  return window.localStorage
}

function makeConversation(id = "conv-1"): PersistedConversation {
  return {
    id,
    title: "Test conversation",
    created_at: "2026-07-30T00:00:00Z",
    updated_at: "2026-07-30T00:00:00Z",
    messages: [
      {
        id: "msg-1",
        role: "user",
        content: "Hello",
        timestamp: "2026-07-30T00:00:00Z",
      },
      {
        id: "msg-2",
        role: "assistant",
        content: "Hi there!",
        timestamp: "2026-07-30T00:00:01Z",
      },
    ],
  }
}

beforeEach(() => {
  storage().clear()
  useChatStore.setState({
    currentConversationId: null,
    panelOpen: false,
    conversations: [],
    sessionMap: {},
    sessionStatus: {},
    _hasHydrated: false,
  })
})

describe("chat-store persistence", () => {
  it("restores conversations from localStorage after rehydration", async () => {
    const conv = makeConversation()
    useChatStore.getState().addConversation(conv)
    useChatStore.getState().setCurrentConversationId("conv-1")
    useChatStore.getState().setSessionId("conv-1", "ses-abc")

    const snapshot = storage().getItem(STORAGE_KEY)
    expect(snapshot).not.toBeNull()

    useChatStore.setState({
      currentConversationId: null,
      conversations: [],
      sessionMap: {},
      sessionStatus: {},
      _hasHydrated: false,
    })

    storage().setItem(STORAGE_KEY, snapshot!)

    await useChatStore.persist.rehydrate()

    const state = useChatStore.getState()
    expect(state.conversations).toHaveLength(1)
    expect(state.conversations[0]!.id).toBe("conv-1")
    expect(state.conversations[0]!.messages).toHaveLength(2)
    expect(state.currentConversationId).toBe("conv-1")
    expect(state.sessionMap["conv-1"]).toBe("ses-abc")
  })

  it("does not persist ephemeral UI state (panelOpen, sessionStatus)", () => {
    useChatStore.getState().setPanelOpen(true)
    useChatStore.getState().setSessionStatus("conv-1", "connected")

    const raw = JSON.parse(storage().getItem(STORAGE_KEY) ?? "{}")
    expect(raw.state).not.toHaveProperty("panelOpen")
    expect(raw.state).not.toHaveProperty("sessionStatus")
  })

  it("falls back to defaults when localStorage is corrupt", async () => {
    storage().setItem(STORAGE_KEY, "NOT_JSON{{{")

    useChatStore.setState({
      currentConversationId: null,
      conversations: [],
      sessionMap: {},
      _hasHydrated: false,
    })

    await useChatStore.persist.rehydrate()

    const state = useChatStore.getState()
    expect(state.conversations).toEqual([])
    expect(state.currentConversationId).toBeNull()
  })

  it("includes version field in persisted data", () => {
    useChatStore.getState().addConversation(makeConversation())

    const raw = JSON.parse(storage().getItem(STORAGE_KEY) ?? "{}")
    expect(raw.version).toBe(1)
  })

  it("migrates v0 data that included panelOpen", async () => {
    const v0Data = {
      state: {
        currentConversationId: "conv-old",
        panelOpen: true,
        conversations: [makeConversation("conv-old")],
        sessionMap: {},
      },
      version: 0,
    }

    useChatStore.setState({
      currentConversationId: null,
      conversations: [],
      sessionMap: {},
      _hasHydrated: false,
    })

    storage().setItem(STORAGE_KEY, JSON.stringify(v0Data))

    await useChatStore.persist.rehydrate()

    const state = useChatStore.getState()
    expect(state.conversations).toHaveLength(1)
    expect(state.currentConversationId).toBe("conv-old")
    expect(state.panelOpen).toBe(false)
  })
})
