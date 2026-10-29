import { create } from "zustand"
import { persist } from "zustand/middleware"

export interface PersistedMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
  // A failed turn: `content` holds the user-facing error text (shown in the chat). Marked so it is
  // excluded from context replay — a restored session must not be told the assistant "said" an
  // error (see summarizeConversation).
  error?: boolean
  toolResults?: { name: string; result: string; collapsed: boolean }[]
  proposedAction?: {
    action: string
    description: string
    params: Record<string, unknown>
    jobType?: "sdg" | "training" | "eval" | "serve"
    endpoint?: string
    config?: Record<string, unknown>
    parentJobId?: string
    recipe?: string
  } | null
  optionCards?: { title: string; description: string; value: string }[]
  selectedOptionValue?: string
  phase?: string
  streamStartedAt?: number
  // Encoded model selection (provider::modelID) that produced this assistant turn,
  // captured from the send-time selection. Absent on legacy messages and user turns.
  model?: string
  // Set on a job-completion continuation turn so the plan deriver can mark the phase
  // terminal even when the driver never emits a "complete" signal_phase (see ChatMessage).
  terminal?: boolean
}

export interface PersistedConversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: PersistedMessage[]
  // Encoded model selection this conversation currently uses (what the next send
  // uses, shown in the header). Set on first send; editable mid-conversation.
  // Absent on legacy conversations (falls back to the global default).
  model?: string
}

export type SessionStatus = "unknown" | "connected" | "reconnecting" | "rebuilt" | "restored"

// Transient in-flight attach progress, keyed by conversation so it survives the keyed
// ChatContent remount on conversation switch (not persisted — it reflects live uploads).
export interface AttachStatus {
  id: number
  message: string
}

interface ChatStoreState {
  currentConversationId: string | null
  panelOpen: boolean
  panelWidth: number
  conversations: PersistedConversation[]
  sessionMap: Record<string, string>
  sessionStatus: Record<string, SessionStatus>
  notifiedJobs: Record<string, string[]>
  jobInFlight: Record<string, boolean>
  // Live requests only: shared by every chat view, but never persisted across reloads.
  activeRequests: Record<string, boolean>
  requestErrors: Record<string, string>
  drafts: Record<string, string>
  attachStatuses: Record<string, AttachStatus[]>
  pendingReferences: Record<string, string[]>
  _hasHydrated: boolean
  setCurrentConversationId: (id: string | null) => void
  setPanelOpen: (open: boolean) => void
  setPanelWidth: (width: number) => void
  togglePanel: () => void
  addConversation: (conv: PersistedConversation) => void
  deleteConversation: (id: string) => void
  replaceAllConversations: (conv: PersistedConversation) => void
  updateConversationTitle: (id: string, title: string) => void
  setConversationModel: (conversationId: string, model: string) => void
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
  tryStartRequest: (conversationId: string) => boolean
  finishRequest: (conversationId: string) => void
  setRequestError: (conversationId: string, error: string | null) => void
  setDraft: (conversationId: string, text: string) => void
  getDraft: (conversationId: string) => string
  addAttachStatus: (conversationId: string, status: AttachStatus) => void
  removeAttachStatus: (conversationId: string, id: number) => void
  enqueueReference: (conversationId: string, text: string) => void
  shiftReference: (conversationId: string) => void
}

export const useChatStore = create<ChatStoreState>()(
  persist(
    (set, get) => ({
      currentConversationId: null,
      panelOpen: false,
      panelWidth: 360,
      conversations: [],
      sessionMap: {},
      sessionStatus: {},
      notifiedJobs: {},
      jobInFlight: {},
      activeRequests: {},
      requestErrors: {},
      drafts: {},
      attachStatuses: {},
      pendingReferences: {},
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
          const { [id]: _ar, ...restActiveRequests } = s.activeRequests  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _re, ...restRequestErrors } = s.requestErrors  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _sd, ...restDrafts } = s.drafts  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _as, ...restAttachStatuses } = s.attachStatuses  // eslint-disable-line @typescript-eslint/no-unused-vars
          const { [id]: _pr, ...restPendingReferences } = s.pendingReferences  // eslint-disable-line @typescript-eslint/no-unused-vars
          return {
            conversations: remaining,
            sessionMap: restSessionMap,
            sessionStatus: restSessionStatus,
            notifiedJobs: restNotifiedJobs,
            jobInFlight: restJobInFlight,
            activeRequests: restActiveRequests,
            requestErrors: restRequestErrors,
            drafts: restDrafts,
            attachStatuses: restAttachStatuses,
            pendingReferences: restPendingReferences,
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
      setConversationModel: (conversationId, model) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId ? { ...c, model } : c,
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
      tryStartRequest: (conversationId) => {
        if (get().activeRequests[conversationId]) return false
        set((s) => {
          const { [conversationId]: _error, ...requestErrors } = s.requestErrors  // eslint-disable-line @typescript-eslint/no-unused-vars
          return { activeRequests: { ...s.activeRequests, [conversationId]: true }, requestErrors }
        })
        return true
      },
      finishRequest: (conversationId) =>
        set((s) => {
          const { [conversationId]: _removed, ...rest } = s.activeRequests  // eslint-disable-line @typescript-eslint/no-unused-vars
          return { activeRequests: rest }
        }),
      setRequestError: (conversationId, error) =>
        set((s) => {
          if (error !== null) return { requestErrors: { ...s.requestErrors, [conversationId]: error } }
          const { [conversationId]: _removed, ...requestErrors } = s.requestErrors  // eslint-disable-line @typescript-eslint/no-unused-vars
          return { requestErrors }
        }),
      setDraft: (conversationId, text) =>
        set((s) => ({
          drafts: text
            ? { ...s.drafts, [conversationId]: text }
            : Object.fromEntries(Object.entries(s.drafts).filter(([k]) => k !== conversationId)),
        })),
      getDraft: (conversationId) => get().drafts[conversationId] ?? "",
      addAttachStatus: (conversationId, status) =>
        set((s) => ({
          attachStatuses: {
            ...s.attachStatuses,
            [conversationId]: [...(s.attachStatuses[conversationId] ?? []), status],
          },
        })),
      removeAttachStatus: (conversationId, id) =>
        set((s) => {
          const next = (s.attachStatuses[conversationId] ?? []).filter((x) => x.id !== id)
          if (next.length === 0) {
            const { [conversationId]: _drop, ...rest } = s.attachStatuses  // eslint-disable-line @typescript-eslint/no-unused-vars
            return { attachStatuses: rest }
          }
          return { attachStatuses: { ...s.attachStatuses, [conversationId]: next } }
        }),
      enqueueReference: (conversationId, text) =>
        set((s) => ({
          pendingReferences: {
            ...s.pendingReferences,
            [conversationId]: [...(s.pendingReferences[conversationId] ?? []), text],
          },
        })),
      shiftReference: (conversationId) =>
        set((s) => {
          const next = (s.pendingReferences[conversationId] ?? []).slice(1)
          if (next.length === 0) {
            const { [conversationId]: _drop, ...rest } = s.pendingReferences  // eslint-disable-line @typescript-eslint/no-unused-vars
            return { pendingReferences: rest }
          }
          return { pendingReferences: { ...s.pendingReferences, [conversationId]: next } }
        }),
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
