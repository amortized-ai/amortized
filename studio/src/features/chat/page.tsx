import { useCallback, useState, useEffect, useMemo } from "react"
import { useNavigate } from "react-router"
import { useChat } from "./hooks/use-chat"
import { useChatStore } from "@/stores/chat-store"
import { useSettingsStore } from "@/stores/settings-store"
import { MessageList } from "./components/message-list"
import { ChatInput } from "./components/chat-input"
import { ConversationList } from "./components/conversation-list"
import { PlanProgress } from "./components/plan-progress"
import { DeleteConversationDialog } from "./components/delete-conversation-dialog"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle, Plus, Bot } from "lucide-react"
import { PROVIDER_CATALOG, encodeModelSelection } from "./models"
import { clearConversationSession } from "@/lib/api-client"

import { derivePhasePlan } from "./utils/derive-plan-steps"

function ChatContent() {
  const navigate = useNavigate()
  const {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    chatState,
    confirmAction,
    rejectAction,
  } = useChat()

  const phasePlan = useMemo(
    () => derivePhasePlan(messages),
    [messages],
  )

  const handleOptionSelect = useCallback(
    (value: string) => {
      if (value.startsWith("__nav:")) {
        navigate(value.slice(6))
        return
      }
      const lastAssistant = [...messages].reverse().find(
        (m) => m.role === "assistant",
      )
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
      <MessageList
        messages={messages}
        onOptionSelect={handleOptionSelect}
        onConfirmAction={confirmAction}
        onRejectAction={rejectAction}
      />
      <ChatInput
        onSend={(msg) => void sendMessage(msg)}
        disabled={isStreaming}
      />
      {chatState === "error" && (
        <Alert variant="destructive" className="rounded-none border-x-0 border-b-0">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Chat error.</strong> Make sure the agent service is running and reachable.
          </AlertDescription>
        </Alert>
      )}
    </>
  )
}

export default function ChatPage() {
  const {
    currentConversationId,
    setCurrentConversationId,
    conversations,
    addConversation,
    deleteConversation,
    replaceAllConversations,
    updateConversationTitle,
    _hasHydrated,
  } = useChatStore()

  const { chatModelSelection, setChatModelSelection, enabledProviders } = useSettingsStore()

  const activeProviders = useMemo(() => {
    return Object.entries(PROVIDER_CATALOG)
      .filter(([id]) => enabledProviders.includes(id))
      .map(([id, info]) => ({ providerID: id, ...info }))
  }, [enabledProviders])

  useEffect(() => {
    const isValid = activeProviders.some((p) =>
      p.models.some((m) => encodeModelSelection(m.providerID, m.modelID) === chatModelSelection)
    )
    if (!isValid && activeProviders.length > 0 && activeProviders[0]!.models.length > 0) {
      const first = activeProviders[0]!.models[0]!
      setChatModelSelection(encodeModelSelection(first.providerID, first.modelID))
    }
  }, [activeProviders, chatModelSelection, setChatModelSelection])

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<{ id: string; title: string } | null>(null)

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
    if (!currentConversationId || !conversations.some((c) => c.id === currentConversationId)) {
      setCurrentConversationId(conversations[0]!.id)
    }
  }, [_hasHydrated, conversations, currentConversationId, addConversation, setCurrentConversationId])

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
  }, [addConversation, setCurrentConversationId])

  const handleSelectConversation = useCallback(
    (id: string) => {
      setCurrentConversationId(id)
    },
    [setCurrentConversationId],
  )

  const handleDeleteConversation = useCallback(
    (id: string) => {
      const conv = conversations.find(c => c.id === id)
      setConversationToDelete({ id, title: conv?.title || "this conversation" })
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
  }, [conversationToDelete, deleteConversation, replaceAllConversations, conversations])

  const cancelDelete = useCallback(() => {
    setDeleteDialogOpen(false)
    setConversationToDelete(null)
  }, [])

  return (
    <div className="-m-4 flex h-[calc(100vh-3rem)] overflow-hidden">
      <ConversationList
        conversations={conversations}
        activeId={currentConversationId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onDelete={handleDeleteConversation}
        onRename={updateConversationTitle}
      />
      <div className="flex flex-1 flex-col min-h-0 bg-accent/20">
        <div className="flex items-center justify-between border-b bg-background/80 backdrop-blur-sm px-4 h-12">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#ee0000]">
              <Bot className="h-3 w-3 text-white" />
            </div>
            <span className="text-sm font-semibold">Morty</span>
            <Select value={chatModelSelection} onValueChange={setChatModelSelection}>
              <SelectTrigger className="h-7 w-[220px] text-xs" data-testid="chat-model-select-page">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {activeProviders.map((provider) => (
                  <SelectGroup key={provider.providerID}>
                    <SelectLabel className="text-xs text-muted-foreground">{provider.label}</SelectLabel>
                    {provider.models.map((m) => {
                      const value = encodeModelSelection(m.providerID, m.modelID)
                      return (
                        <SelectItem key={value} value={value} className="text-xs">
                          {m.label}
                        </SelectItem>
                      )
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button size="sm" variant="outline" onClick={handleNewConversation} className="rounded-lg transition-all duration-300 hover:shadow-sm hover:-translate-y-px">
            <Plus className="mr-1 h-3.5 w-3.5" />
            New
          </Button>
        </div>
        <ChatContent key={currentConversationId} />
      </div>

      <DeleteConversationDialog
        open={deleteDialogOpen}
        conversationTitle={conversationToDelete?.title || "this conversation"}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </div>
  )
}
