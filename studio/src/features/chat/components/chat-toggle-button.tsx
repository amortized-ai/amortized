import { MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useChatStore } from "@/stores/chat-store"
import { ChatPanel } from "./chat-panel"

export function ChatToggleButton() {
  const { togglePanel } = useChatStore()

  return (
    <>
      <Button
        size="icon"
        className="fixed bottom-6 right-6 z-50 h-12 w-12 rounded-full shadow-lg"
        onClick={togglePanel}
        data-testid="chat-toggle-button"
        aria-label="Toggle chat"
      >
        <MessageSquare className="h-5 w-5" />
      </Button>
      <ChatPanel />
    </>
  )
}
