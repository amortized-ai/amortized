import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Plus, Search, MessageSquare, MessagesSquare, Trash2, Pencil } from "lucide-react"
import { EmptyState } from "@/components/empty-state"
import { ConversationSearchDialog } from "./conversation-search-dialog"
import type { Conversation } from "../types"

interface ConversationListProps {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete?: (id: string) => void
  onRename?: (id: string, title: string) => void
}

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
}: ConversationListProps) {
  const [searchOpen, setSearchOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState("")
  const editRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editingId) {
      editRef.current?.focus()
      editRef.current?.select()
    }
  }, [editingId])

  function startEditing(conv: Conversation) {
    setEditingId(conv.id)
    setEditValue(conv.title || "")
  }

  function commitEdit() {
    if (editingId && editValue.trim() && onRename) {
      onRename(editingId, editValue.trim())
    }
    setEditingId(null)
    setEditValue("")
  }

  function cancelEdit() {
    setEditingId(null)
    setEditValue("")
  }

  return (
    <div className="flex h-full w-80 flex-col border-r bg-muted/30" style={{ width: '320px' }}>
      <div className="flex items-center justify-between border-b bg-background/80 backdrop-blur-sm px-3 h-12">
        <span className="text-sm font-semibold">Conversations</span>
        <div className="flex items-center gap-1">
          {conversations.length > 0 && (
            <Button
              size="icon"
              variant="ghost"
              onClick={() => setSearchOpen(true)}
              className="h-7 w-7 rounded-lg transition-all duration-300 hover:shadow-sm"
              aria-label="Search conversations"
            >
              <Search className="h-4 w-4" />
            </Button>
          )}
          <Button size="icon" variant="ghost" onClick={onNew} className="h-7 w-7 rounded-lg transition-all duration-300 hover:shadow-sm" aria-label="New conversation">
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="flex flex-col gap-0.5 p-2" style={{ width: '100%' }}>
          {conversations.length === 0 && (
            <EmptyState
              icon={MessagesSquare}
              title="No conversations"
              description="Start a new conversation to begin."
            />
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={cn(
                "group flex items-center gap-2 rounded-xl px-2.5 py-2.5 transition-all duration-300",
                "hover:bg-background hover:shadow-sm hover:-translate-y-px",
                activeId === conv.id ? "bg-background shadow-sm border border-border/50 border-l-2 border-l-primary" : "border border-transparent"
              )}
            >
              <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

              {editingId === conv.id ? (
                <div className="flex-1 flex items-center gap-1 min-w-0">
                  <input
                    ref={editRef}
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitEdit()
                      if (e.key === "Escape") cancelEdit()
                    }}
                    onBlur={commitEdit}
                    className="flex-1 min-w-0 bg-transparent text-sm font-medium outline-none border-b border-primary/40 dark:border-primary/50 py-0.5"
                  />
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => onSelect(conv.id)}
                  onDoubleClick={() => startEditing(conv)}
                  className="flex-1 text-left overflow-hidden min-w-0"
                  title="Double-click to rename"
                >
                  <span className="block truncate text-sm font-medium">
                    {conv.title || "Untitled"}
                  </span>
                </button>
              )}

              {editingId !== conv.id && (
                <div className="flex items-center gap-0.5 shrink-0">
                  {onRename && (
                    <button
                      type="button"
                      aria-label={`Rename ${conv.title || "conversation"}`}
                      className="rounded p-1.5 text-muted-foreground/40 opacity-0 transition-all duration-200 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                      onClick={(e) => {
                        e.stopPropagation()
                        startEditing(conv)
                      }}
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                  )}
                  <button
                    type="button"
                    aria-label={`Delete ${conv.title || "conversation"}`}
                    className="rounded p-1.5 text-muted-foreground/40 opacity-0 transition-all duration-200 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-rh-danger-light/50 hover:text-rh-danger dark:hover:bg-rh-danger-dark/30 dark:hover:text-rh-danger focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                    onClick={(e) => {
                      e.stopPropagation()
                      if (onDelete) {
                        onDelete(conv.id)
                      }
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <ConversationSearchDialog
        open={searchOpen}
        onOpenChange={setSearchOpen}
        conversations={conversations}
        onSelect={onSelect}
      />
    </div>
  )
}
