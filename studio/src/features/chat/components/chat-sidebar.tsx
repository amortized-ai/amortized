import { useCallback, useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router"
import { Bot, X, Plus, ChevronDown, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useChatStore } from "@/stores/chat-store"
import { useChat } from "../hooks/use-chat"
import { SessionStatusBanner } from "./session-status-banner"
import { MessageList } from "./message-list"
import { ChatInput } from "./chat-input"
import { PlanProgress } from "./plan-progress"
import { DeleteConversationDialog } from "./delete-conversation-dialog"
import { derivePhasePlan } from "../utils/derive-plan-steps"
import { clearConversationSession } from "@/lib/api-client"

export function ChatSidebar() {
  const navigate = useNavigate()
  const {
    currentConversationId,
    setCurrentConversationId,
    conversations,
    addConversation,
    deleteConversation,
    replaceAllConversations,
    setPanelOpen,
    _hasHydrated,
  } = useChatStore()

  const {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    confirmAction,
    rejectAction,
  } = useChat()

  const sessionStatus = useChatStore(
    (s) => s.sessionStatus[s.currentConversationId ?? ""] ?? "unknown",
  )

  const phasePlan = useMemo(() => derivePhasePlan(messages), [messages])

  const [convListOpen, setConvListOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<{
    id: string
    title: string
  } | null>(null)

  const currentConversation = conversations.find(
    (c) => c.id === currentConversationId,
  )

  const handleSend = useCallback(
    (value: string) => {
      if (value.startsWith("__nav:")) {
        navigate(value.slice(6))
        return
      }
      const lastAssistant = [...messages]
        .reverse()
        .find((m) => m.role === "assistant")
      if (lastAssistant) {
        selectOption(lastAssistant.id, value)
      }
      void sendMessage(value)
    },
    [sendMessage, selectOption, messages, navigate],
  )

  const handleNewConversation = useCallback(() => {
    const now = new Date().toISOString()
    const convId = `conv-${Date.now()}`
    addConversation({
      id: convId,
      title: "New conversation",
      created_at: now,
      updated_at: now,
      messages: [],
    })
    setCurrentConversationId(convId)
    setConvListOpen(false)
  }, [addConversation, setCurrentConversationId])

  const handleDeleteConversation = useCallback(
    (id: string) => {
      const conv = conversations.find((c) => c.id === id)
      setConversationToDelete({
        id,
        title: conv?.title || "this conversation",
      })
      setDeleteDialogOpen(true)
    },
    [conversations],
  )

  const confirmDelete = useCallback(() => {
    if (!conversationToDelete) return
    clearConversationSession(conversationToDelete.id)

    if (conversations.length <= 1) {
      const now = new Date().toISOString()
      const convId = `conv-${Date.now()}`
      replaceAllConversations({
        id: convId,
        title: "New conversation",
        created_at: now,
        updated_at: now,
        messages: [],
      })
    } else {
      deleteConversation(conversationToDelete.id)
    }

    setDeleteDialogOpen(false)
    setConversationToDelete(null)
  }, [
    conversationToDelete,
    deleteConversation,
    replaceAllConversations,
    conversations,
  ])

  const cancelDelete = useCallback(() => {
    setDeleteDialogOpen(false)
    setConversationToDelete(null)
  }, [])

  useEffect(() => {
    if (!_hasHydrated) return
    if (conversations.length === 0) {
      const now = new Date().toISOString()
      const convId = `conv-${Date.now()}`
      addConversation({
        id: convId,
        title: "New conversation",
        created_at: now,
        updated_at: now,
        messages: [],
      })
      setCurrentConversationId(convId)
      return
    }
    if (
      !currentConversationId ||
      !conversations.some((c) => c.id === currentConversationId)
    ) {
      setCurrentConversationId(conversations[0]!.id)
    }
  }, [
    _hasHydrated,
    conversations,
    currentConversationId,
    addConversation,
    setCurrentConversationId,
  ])

  if (!_hasHydrated) return null

  return (
    <div className="flex h-full flex-col border-l bg-background">
      {/* Header */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b px-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#ee0000]">
            <Bot className="h-3 w-3 text-white" />
          </div>
          <button
            onClick={() => setConvListOpen(!convListOpen)}
            className="flex items-center gap-1 min-w-0 text-sm font-medium hover:text-foreground/80 transition-colors"
          >
            <span className="truncate max-w-[140px]">
              {currentConversation?.title ?? "Chat"}
            </span>
            <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
          </button>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleNewConversation}
            aria-label="New conversation"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setPanelOpen(false)}
            aria-label="Close chat"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Conversation dropdown */}
      {convListOpen && (
        <div className="border-b bg-muted/30 max-h-48 overflow-y-auto">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group flex items-center justify-between px-3 py-2 text-xs cursor-pointer hover:bg-accent transition-colors ${
                conv.id === currentConversationId
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground"
              }`}
            >
              <button
                className="truncate text-left flex-1 min-w-0"
                onClick={() => {
                  setCurrentConversationId(conv.id)
                  setConvListOpen(false)
                }}
              >
                {conv.title}
              </button>
              <button
                className="shrink-0 ml-2 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity"
                onClick={(e) => {
                  e.stopPropagation()
                  handleDeleteConversation(conv.id)
                }}
                aria-label={`Delete ${conv.title}`}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Plan progress */}
      <PlanProgress plan={phasePlan} />

      {/* Session status */}
      <SessionStatusBanner
        status={sessionStatus}
        onDismiss={() => {
          const id = useChatStore.getState().currentConversationId
          if (id) useChatStore.getState().setSessionStatus(id, "connected")
        }}
      />

      {/* Messages */}
      <MessageList
        messages={messages}
        onOptionSelect={handleSend}
        onConfirmAction={confirmAction}
        onRejectAction={rejectAction}
      />

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isStreaming} />

      <DeleteConversationDialog
        open={deleteDialogOpen}
        conversationTitle={conversationToDelete?.title || "this conversation"}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </div>
  )
}
