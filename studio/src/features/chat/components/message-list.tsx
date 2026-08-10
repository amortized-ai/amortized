import { useEffect, useRef } from "react"
import type { ChatMessage } from "../types"
import { MessageBubble } from "./message-bubble"
import { ChatWelcome } from "./chat-welcome"

interface MessageListProps {
  messages: ChatMessage[]
  onOptionSelect?: (value: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
}

export function MessageList({
  messages,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
}: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  const lastContent = messages[messages.length - 1]?.content ?? ""

  useEffect(() => {
    const t = setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
    }, 300)
    return () => clearTimeout(t)
  }, [messages.length, lastContent])

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      {messages.length === 0 ? (
        <ChatWelcome onPromptClick={onOptionSelect || (() => {})} />
      ) : (
        <div className="space-y-4 p-4">
          {messages.map((message, i) => (
            <MessageBubble
              key={message.id}
              message={message}
              isLatest={i === messages.length - 1}

              onOptionSelect={onOptionSelect}
              onConfirmAction={onConfirmAction}
              onRejectAction={onRejectAction}
            />
          ))}
          <div ref={endRef} />
        </div>
      )}
    </div>
  )
}
