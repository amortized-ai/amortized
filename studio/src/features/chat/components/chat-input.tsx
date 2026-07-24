import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Send } from "lucide-react"

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSubmit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  function handleInput() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }

  return (
    <div className="border-t bg-background/80 backdrop-blur-sm p-4" data-tutorial="chat-input">
      <div className="mx-auto max-w-3xl flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Type a message..."
          disabled={disabled}
          rows={1}
          aria-label="Message input"
          className="flex-1 resize-none rounded-xl border border-input bg-card px-4 py-3 text-sm shadow-sm placeholder:text-muted-foreground/60 hover:shadow-md hover:border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20 focus-visible:border-primary/40 focus-visible:shadow-md dark:focus-visible:ring-primary/10 dark:focus-visible:border-primary/50 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200"
        />
        <Button
          size="icon"
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
          className="h-11 w-11 rounded-xl shrink-0 transition-all duration-200 hover:scale-105 hover:shadow-md"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
