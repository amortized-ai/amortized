import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useNavigate } from "react-router"
import { Bot, X, Plus, ChevronDown, Trash2, GripVertical } from "lucide-react"
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

const MIN_WIDTH = 320
const MAX_WIDTH = 700

function SidebarChatContent() {
  const navigate = useNavigate()
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

  return (
    <>
      <PlanProgress plan={phasePlan} />
      <SessionStatusBanner
        status={sessionStatus}
        onDismiss={() => {
          const id = useChatStore.getState().currentConversationId
          if (id) useChatStore.getState().setSessionStatus(id, "connected")
        }}
      />
      <MessageList
        messages={messages}
        onOptionSelect={handleSend}
        onConfirmAction={confirmAction}
        onRejectAction={rejectAction}
      />
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </>
  )
}

export function ChatSidebar() {
  const {
    currentConversationId,
    setCurrentConversationId,
    conversations,
    addConversation,
    deleteConversation,
    replaceAllConversations,
    setPanelOpen,
    setPanelWidth,
    _hasHydrated,
  } = useChatStore()

  const isDragging = useRef(false)

  const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    const startX = e.clientX
    const startWidth = useChatStore.getState().panelWidth

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return
      const delta = startX - ev.clientX
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth + delta))
      setPanelWidth(newWidth)
    }

    const onMouseUp = () => {
      isDragging.current = false
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [setPanelWidth])

  const [convListOpen, setConvListOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<{
    id: string
    title: string
  } | null>(null)

  const currentConversation = conversations.find(
    (c) => c.id === currentConversationId,
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

  return (
    <div className="flex h-full">
      {/* Resize handle */}
      <div
        onMouseDown={onResizeMouseDown}
        className="group flex w-1.5 shrink-0 cursor-col-resize items-center justify-center border-l hover:bg-primary/10 active:bg-primary/20 transition-colors"
      >
        <GripVertical className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
      </div>
      {/* Chat content */}
      <div className="flex flex-1 min-w-0 flex-col bg-background">
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

        {/* Keyed on conversationId so useChat remounts and loads correct messages */}
        <SidebarChatContent key={currentConversationId ?? "none"} />

        <DeleteConversationDialog
          open={deleteDialogOpen}
          conversationTitle={conversationToDelete?.title || "this conversation"}
          onConfirm={confirmDelete}
          onCancel={cancelDelete}
        />
      </div>
    </div>
  )
}
