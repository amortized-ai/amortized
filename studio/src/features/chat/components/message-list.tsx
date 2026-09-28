import { useEffect, useRef } from "react"
import type { ChatMessage } from "../types"
import { MessageBubble } from "./message-bubble"
import { ChatWelcome } from "./chat-welcome"

interface MessageListProps {
  messages: ChatMessage[]
  thinkingStep?: string | null
  isStreaming?: boolean
  onOptionSelect?: (value: string, messageId?: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
  onJobComplete?: (jobId: string, jobType: string, status: string) => void
}

export function MessageList({
  messages,
  thinkingStep,
  isStreaming,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
  onJobComplete,
}: MessageListProps) {
  // `messages` is already derived from the store by useChat (single source of truth), so
  // no separate store-read fallback is needed here.
  const endRef = useRef<HTMLDivElement>(null)

  const lastContent = messages[messages.length - 1]?.content ?? ""

  useEffect(() => {
    const t = setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
    }, 300)
    return () => clearTimeout(t)
  }, [messages.length, lastContent, thinkingStep])

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      {messages.length === 0 && !isStreaming ? (
        <ChatWelcome onPromptClick={onOptionSelect || (() => {})} />
      ) : (
        <div className="space-y-4 p-4">
          {messages.map((message, i) => (
            <MessageBubble
              key={message.id}
              message={message}
              isLatest={i === messages.length - 1}
              thinkingStep={i === messages.length - 1 ? thinkingStep : undefined}
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
