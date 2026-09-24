import { useState } from "react"
import { CheckCircle } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"
import { useChatStore } from "@/stores/chat-store"
import { markRunComplete } from "@/lib/api-client"

/**
 * Records a human-declared completion for a test run. The platform has no
 * workflow-completion signal (the orchestrator loops indefinitely), so the
 * tester marks the boundary here; it defines turns-to-complete / total-tokens
 * for the monitor metrics.
 */
export function MarkCompleteButton() {
  const conversationId = useChatStore((s) => s.currentConversationId)
  const [busy, setBusy] = useState(false)

  async function mark(outcome: "success" | "gave_up") {
    if (!conversationId || busy) return
    setBusy(true)
    const ok = await markRunComplete(conversationId, outcome)
    setBusy(false)
    if (ok) {
      toast.success(
        outcome === "success" ? "Run marked complete" : "Run marked as gave-up",
      )
    } else {
      toast.error("Could not record completion")
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1 text-xs"
          disabled={!conversationId || busy}
        >
          <CheckCircle className="h-3.5 w-3.5" />
          Mark complete
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onSelect={() => void mark("success")}>
          Completed — success
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => void mark("gave_up")}>
          Ended — gave up
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
