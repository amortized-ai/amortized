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
import { Plus, Bot } from "lucide-react"
import { PROVIDER_CATALOG, encodeModelSelection } from "./models"
import { useProviderStatus } from "./api/use-providers"
import { clearConversationSession } from "@/lib/api-client"

import { derivePlan } from "./utils/derive-plan-steps"

function ChatContent() {
  const navigate = useNavigate()
  const {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    chatState,
    error,
    thinkingStep,
    confirmAction,
    rejectAction,
    notifyJobComplete,
  } = useChat()

  const phasePlan = useMemo(
    () => derivePlan(messages),
    [messages],
  )

  const handleOptionSelect = useCallback(
    (value: string, messageId?: string) => {
      if (value.startsWith("__nav:")) {
        navigate(value.slice(6))
        return
      }
      if (messageId) {
        selectOption(messageId, value)
      }
      void sendMessage(value)
    },
    [sendMessage, selectOption, navigate],
  )

  return (
    <>
      <PlanProgress plan={phasePlan} />
      <MessageList
        messages={messages}
        thinkingStep={thinkingStep}
        isStreaming={isStreaming}
        onOptionSelect={handleOptionSelect}
        onConfirmAction={confirmAction}
        onRejectAction={rejectAction}
        onJobComplete={notifyJobComplete}
      />
      <ChatInput
        onSend={(msg) => void sendMessage(msg)}
        disabled={isStreaming}
      />
      {chatState === "error" && (
        <Alert variant="destructive">
          <AlertDescription>
            {error || "Something went wrong. Make sure the agent service is running and reachable."}
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

  const { chatModelSelection, setChatModelSelection, enabledProviders, setEnabledProviders } =
    useSettingsStore()
  const { connectedProviders, providers: dynamicProviders } = useProviderStatus()

  // Build provider list: connected+catalog intersection only (avoids the hundreds of built-in
  // opencode providers that have no models and aren't relevant to this deploy).
  const allProviders = useMemo(() => {
    if (dynamicProviders.length > 0) {
      return dynamicProviders
        .filter((p) => (p.id in PROVIDER_CATALOG || connectedProviders.has(p.id)) && p.models.length > 0)
        .map((p) => ({
          providerID: p.id,
          label: PROVIDER_CATALOG[p.id]?.label ?? p.name,
          requiresApiKey: PROVIDER_CATALOG[p.id]?.requiresApiKey ?? false,
          models: p.models,
        }))
    }
    return Object.entries(PROVIDER_CATALOG).map(([id, info]) => ({ providerID: id, ...info }))
  }, [dynamicProviders, connectedProviders])

  const activeProviders = useMemo(
    () => allProviders.filter((p) => enabledProviders.includes(p.providerID)),
    [allProviders, enabledProviders],
  )

  // Providers the backend actually has credentials for, limited to ones we render.
  const connectedKnownProviders = useMemo(
    () => allProviders.map((p) => p.providerID).filter((id) => connectedProviders.has(id)),
    [allProviders, connectedProviders],
  )

  // Providers the user can actually pick. When connectivity is known, restrict to
  // connected ones so a disconnected-but-enabled provider can't be selected (its turns
  // would 500); otherwise (status unknown / offline) fall back to all enabled providers.
  const usableProviders = useMemo(
    () =>
      connectedKnownProviders.length > 0
        ? activeProviders.filter((p) => connectedProviders.has(p.providerID))
        : activeProviders,
    [activeProviders, connectedKnownProviders, connectedProviders],
  )

  // If the backend reports connected providers but none of the enabled ones are actually
  // connected, enable the connected ones. Without this, a deploy whose only credentialed
  // provider differs from the hard-coded default (e.g. OpenAI-only while the default is
  // Vertex) leaves the user stuck on a provider the agent can't serve → the first turn 500s.
  useEffect(() => {
    if (connectedKnownProviders.length === 0) return
    const anyEnabledConnected = enabledProviders.some((id) => connectedProviders.has(id))
    if (!anyEnabledConnected) {
      setEnabledProviders(connectedKnownProviders)
    }
  }, [connectedKnownProviders, connectedProviders, enabledProviders, setEnabledProviders])

  // Keep the selection on a usable provider (see usableProviders above).
  useEffect(() => {
    const isValid = usableProviders.some((p) =>
      p.models.some((m) => encodeModelSelection(m.providerID, m.modelID) === chatModelSelection)
    )
    const firstWithModels = usableProviders.find((p) => p.models.length > 0)
    if (!isValid && firstWithModels) {
      const first = firstWithModels.models[0]!
      setChatModelSelection(encodeModelSelection(first.providerID, first.modelID))
    }
  }, [usableProviders, chatModelSelection, setChatModelSelection])

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
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-rh-red">
              <Bot className="h-3 w-3 text-white" />
            </div>
            <span className="text-sm font-semibold">Morty</span>
            <Select value={chatModelSelection} onValueChange={setChatModelSelection}>
              <SelectTrigger className="h-7 w-[220px] text-xs" data-testid="chat-model-select-page">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {usableProviders.map((provider) => (
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
        <ChatContent key={currentConversationId ?? "empty"} />
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
