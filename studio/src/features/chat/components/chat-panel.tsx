import { useCallback, useMemo } from "react"
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

export function ChatPanel() {
  const { panelOpen, setPanelOpen } = useChatStore()
  const {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    confirmAction,
    rejectAction,
  } = useChat()

  const sessionStatus = useChatStore(s =>
    s.sessionStatus[s.currentConversationId ?? ""] ?? "unknown"
  )

  const phasePlan = useMemo(
    () => derivePlan(messages),
    [messages],
  )

  const handleOptionSelect = useCallback(
    (value: string) => {
      const lastAssistant = [...messages].reverse().find(
        (m) => m.role === "assistant",
      )
      if (lastAssistant) {
        selectOption(lastAssistant.id, value)
      }
      void sendMessage(value)
    },
    [sendMessage, selectOption, messages],
  )

  return (
    <Sheet open={panelOpen} onOpenChange={setPanelOpen}>
      <SheetContent side="right" className="flex w-[400px] flex-col p-0 sm:max-w-[400px]">
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
          onOptionSelect={handleOptionSelect}
          onConfirmAction={confirmAction}
          onRejectAction={rejectAction}
        />
        <ChatInput onSend={handleOptionSelect} disabled={isStreaming} />
      </SheetContent>
    </Sheet>
  )
}
