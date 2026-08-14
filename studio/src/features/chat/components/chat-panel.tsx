import { useCallback, useMemo } from "react"
import { useNavigate } from "react-router"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { useChatStore } from "@/stores/chat-store"
import { useChat } from "../hooks/use-chat"
import { SessionStatusBanner } from "./session-status-banner"
import { MessageList } from "./message-list"
import { ChatInput } from "./chat-input"
import { PlanProgress } from "./plan-progress"
import { derivePlan } from "../utils/derive-plan-steps"

function ChatPanelContent() {
  const navigate = useNavigate()
  const {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    confirmAction,
    rejectAction,
    notifyJobComplete,
  } = useChat()

  const sessionStatus = useChatStore(s =>
    s.sessionStatus[s.currentConversationId ?? ""] ?? "unknown"
  )

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
      <SheetHeader className="border-b p-4">
        <SheetTitle>Chat</SheetTitle>
      </SheetHeader>
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
        isStreaming={isStreaming}
        onOptionSelect={handleOptionSelect}
        onConfirmAction={confirmAction}
        onRejectAction={rejectAction}
        onJobComplete={notifyJobComplete}
      />
      <ChatInput onSend={handleOptionSelect} disabled={isStreaming} />
    </>
  )
}

export function ChatPanel() {
  const { panelOpen, setPanelOpen, currentConversationId } = useChatStore()

  return (
    <Sheet open={panelOpen} onOpenChange={setPanelOpen}>
      <SheetContent side="right" className="flex w-[400px] flex-col p-0 sm:max-w-[400px]">
        <ChatPanelContent key={currentConversationId ?? "empty"} />
      </SheetContent>
    </Sheet>
  )
}
