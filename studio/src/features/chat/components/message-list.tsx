import { useEffect, useMemo, useRef } from "react"
import type { ChatMessage } from "../types"
import { useChatStore, type PersistedMessage } from "@/stores/chat-store"
import { MessageBubble } from "./message-bubble"
import { ChatWelcome } from "./chat-welcome"

interface MessageListProps {
  messages: ChatMessage[]
  isStreaming?: boolean
  onOptionSelect?: (value: string, messageId?: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
  onJobComplete?: (jobId: string, jobType: string, status: string) => void
}

const EMPTY: PersistedMessage[] = []

function toChat(msgs: PersistedMessage[]): ChatMessage[] {
  return msgs.map((m) => ({
    id: m.id,
    role: m.role,
    content: m.content,
    timestamp: m.timestamp,
    toolResults: m.toolResults ?? [],
    proposedAction: m.proposedAction ?? null,
    optionCards: m.optionCards ?? [],
    selectedOptionValue: m.selectedOptionValue,
    phase: m.phase,
  }))
}

export function MessageList({
  messages,
  isStreaming,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
  onJobComplete,
}: MessageListProps) {
  const rawStoreMessages = useChatStore((s) => {
    const convId = s.currentConversationId
    if (convId) {
      const conv = s.conversations.find((c) => c.id === convId)
      if (conv) return conv.messages.length > 0 ? conv.messages : EMPTY
    }
    return EMPTY
  })
  const storeMessages = useMemo(() => toChat(rawStoreMessages), [rawStoreMessages])
  const displayMessages = messages.length > 0 ? messages : storeMessages

  const endRef = useRef<HTMLDivElement>(null)

  const lastContent = displayMessages[displayMessages.length - 1]?.content ?? ""

  useEffect(() => {
    const t = setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
    }, 300)
    return () => clearTimeout(t)
  }, [displayMessages.length, lastContent])

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      {displayMessages.length === 0 && rawStoreMessages.length === 0 && !isStreaming ? (
        <ChatWelcome onPromptClick={onOptionSelect || (() => {})} />
      ) : (
        <div className="space-y-4 p-4">
          {displayMessages.map((message, i) => (
            <MessageBubble
              key={message.id}
              message={message}
              isLatest={i === displayMessages.length - 1}
              onOptionSelect={onOptionSelect}
              onConfirmAction={onConfirmAction}
              onRejectAction={onRejectAction}
              onJobComplete={onJobComplete}
            />
          ))}
          <div ref={endRef} />
        </div>
      )}
    </div>
  )
}
