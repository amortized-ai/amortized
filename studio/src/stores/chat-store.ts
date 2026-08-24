import { create } from "zustand"
import { persist } from "zustand/middleware"

export interface PersistedMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
  toolResults?: { name: string; result: string; collapsed: boolean }[]
  proposedAction?: { action: string; description: string; params: Record<string, unknown> } | null
  optionCards?: { title: string; description: string; value: string }[]
  selectedOptionValue?: string
  phase?: string
  streamStartedAt?: number
}

export interface PersistedConversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: PersistedMessage[]
}

export type SessionStatus = "unknown" | "connected" | "reconnecting" | "rebuilt" | "restored"

interface ChatStoreState {
  currentConversationId: string | null
  panelOpen: boolean
  panelWidth: number
  conversations: PersistedConversation[]
  sessionMap: Record<string, string>
  sessionStatus: Record<string, SessionStatus>
  notifiedJobs: Record<string, string[]>
  jobInFlight: Record<string, boolean>
  drafts: Record<string, string>
  _hasHydrated: boolean
  setCurrentConversationId: (id: string | null) => void
  setPanelOpen: (open: boolean) => void
  setPanelWidth: (width: number) => void
  togglePanel: () => void
  addConversation: (conv: PersistedConversation) => void
  deleteConversation: (id: string) => void
  replaceAllConversations: (conv: PersistedConversation) => void
  updateConversationTitle: (id: string, title: string) => void
  addMessage: (conversationId: string, message: PersistedMessage) => void
  updateMessage: (conversationId: string, messageId: string, content: string) => void
  getConversationMessages: (conversationId: string) => PersistedMessage[]
  updateMessageFields: (conversationId: string, messageId: string, fields: Partial<PersistedMessage>) => void
  removeMessage: (conversationId: string, messageId: string) => void
  setSessionId: (conversationId: string, sessionId: string) => void
  getSessionId: (conversationId: string) => string | undefined
  clearSessionId: (conversationId: string) => void
  setSessionStatus: (conversationId: string, status: SessionStatus) => void
  getSessionStatus: (conversationId: string) => SessionStatus
  addNotifiedJob: (conversationId: string, jobId: string) => void
  getNotifiedJobs: (conversationId: string) => string[]
  setJobInFlight: (conversationId: string, inFlight: boolean) => void
  getJobInFlight: (conversationId: string) => boolean
  setDraft: (conversationId: string, text: string) => void
  getDraft: (conversationId: string) => string
}

export const useChatStore = create<ChatStoreState>()(
  persist(
    (set, get) => ({
      currentConversationId: null,
      panelOpen: false,
      panelWidth: 400,
      conversations: [],
      sessionMap: {},
      sessionStatus: {},
      notifiedJobs: {},
      jobInFlight: {},
      drafts: {},
      _hasHydrated: false,
      setCurrentConversationId: (id) => set({ currentConversationId: id }),
      setPanelOpen: (open) => set({ panelOpen: open }),
      setPanelWidth: (width) => set({ panelWidth: width }),
      togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
      addConversation: (conv) =>
        set((s) => ({ conversations: [conv, ...s.conversations] })),
      deleteConversation: (id) =>
        set((s) => {
          const remaining = s.conversations.filter((c) => c.id !== id)
          const { [id]: _sid, ...restSessionMap } = s.sessionMap  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _ss, ...restSessionStatus } = s.sessionStatus  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _sn, ...restNotifiedJobs } = s.notifiedJobs  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _jf, ...restJobInFlight } = s.jobInFlight  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _sd, ...restDrafts } = s.drafts  // eslint-disable-line @typescript-eslint/no-unused-vars
          return {
            conversations: remaining,
            sessionMap: restSessionMap,
            sessionStatus: restSessionStatus,
            notifiedJobs: restNotifiedJobs,
            jobInFlight: restJobInFlight,
            drafts: restDrafts,
            currentConversationId:
              s.currentConversationId === id
                ? (remaining[0]?.id ?? null)
                : s.currentConversationId,
          }
        }),
      replaceAllConversations: (conv) =>
        set({ conversations: [conv], currentConversationId: conv.id }),
      updateConversationTitle: (id, title) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, title } : c,
          ),
        })),
      addMessage: (conversationId, message) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? { ...c, messages: [...c.messages, message], updated_at: new Date().toISOString() }
              : c,
          ),
        })),
      updateMessage: (conversationId, messageId, content) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === messageId ? { ...m, content } : m,
                  ),
                }
              : c,
          ),
        })),
      getConversationMessages: (conversationId) => {
        const conv = get().conversations.find((c) => c.id === conversationId)
        return conv?.messages ?? []
      },
      updateMessageFields: (conversationId, messageId, fields) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === messageId ? { ...m, ...fields } : m,
                  ),
                }
              : c,
          ),
        })),
      removeMessage: (conversationId, messageId) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? { ...c, messages: c.messages.filter((m) => m.id !== messageId) }
              : c,
          ),
        })),
      setSessionId: (conversationId, sessionId) =>
        set((s) => ({
          sessionMap: { ...s.sessionMap, [conversationId]: sessionId },
        })),
      getSessionId: (conversationId) => get().sessionMap[conversationId],
      clearSessionId: (conversationId) =>
        set((s) => {
          const { [conversationId]: _removed, ...rest } = s.sessionMap  // eslint-disable-line @typescript-eslint/no-unused-vars
          return { sessionMap: rest }
        }),
      setSessionStatus: (conversationId, status) =>
        set((s) => ({
          sessionStatus: { ...s.sessionStatus, [conversationId]: status },
        })),
      getSessionStatus: (conversationId) =>
        get().sessionStatus[conversationId] ?? "unknown",
      addNotifiedJob: (conversationId, jobId) =>
        set((s) => ({
          notifiedJobs: {
            ...s.notifiedJobs,
            [conversationId]: [...(s.notifiedJobs[conversationId] ?? []), jobId],
          },
        })),
      getNotifiedJobs: (conversationId) =>
        get().notifiedJobs[conversationId] ?? [],
      setJobInFlight: (conversationId, inFlight) =>
        set((s) => ({
          jobInFlight: inFlight
            ? { ...s.jobInFlight, [conversationId]: true }
            : Object.fromEntries(Object.entries(s.jobInFlight).filter(([k]) => k !== conversationId)),
        })),
      getJobInFlight: (conversationId) =>
        get().jobInFlight[conversationId] ?? false,
      setDraft: (conversationId, text) =>
        set((s) => ({
          drafts: text
            ? { ...s.drafts, [conversationId]: text }
            : Object.fromEntries(Object.entries(s.drafts).filter(([k]) => k !== conversationId)),
        })),
      getDraft: (conversationId) => get().drafts[conversationId] ?? "",
    }),
    {
      name: "amortized-chat",
      partialize: (state) => ({
        currentConversationId: state.currentConversationId,
        panelOpen: state.panelOpen,
        panelWidth: state.panelWidth,
        conversations: state.conversations,
        sessionMap: state.sessionMap,
        notifiedJobs: state.notifiedJobs,
        jobInFlight: state.jobInFlight,
        drafts: state.drafts,
      }),
    },
  ),
)

if (useChatStore.persist.hasHydrated()) {
  useChatStore.setState({ _hasHydrated: true })
} else {
  useChatStore.persist.onFinishHydration(() => {
    useChatStore.setState({ _hasHydrated: true })
  })
}
