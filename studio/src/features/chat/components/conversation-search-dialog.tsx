import { useState, useMemo, useEffect, useRef, useCallback } from "react"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { Search, MessageSquare, Quote, CornerDownLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import { useChatStore } from "@/stores/chat-store"
import type { Conversation } from "../types"

interface SearchResult {
  conversation: Conversation
  matchType: "title" | "message"
  snippet: string
}

interface ConversationSearchDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  conversations: Conversation[]
  onSelect: (id: string) => void
}

function getSnippet(content: string, query: string): string {
  const lower = content.toLowerCase()
  const idx = lower.indexOf(query.toLowerCase())
  if (idx === -1) return ""
  const start = Math.max(0, idx - 40)
  const end = Math.min(content.length, idx + query.length + 60)
  let snippet = content.slice(start, end).replace(/\n/g, " ").replace(/\s+/g, " ")
  if (start > 0) snippet = "…" + snippet
  if (end < content.length) snippet = snippet + "…"
  return snippet
}

export function ConversationSearchDialog({
  open,
  onOpenChange,
  conversations,
  onSelect,
}: ConversationSearchDialogProps) {
  const [query, setQuery] = useState("")
  const [selectedIdx, setSelectedIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const storeConversations = useChatStore((s) => s.conversations)

  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset on open
      setQuery("")
      setSelectedIdx(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const results = useMemo<SearchResult[]>(() => {
    const q = query.trim()
    if (!q) return []
    const lower = q.toLowerCase()

    const matches: SearchResult[] = []
    for (const conv of conversations) {
      if ((conv.title || "").toLowerCase().includes(lower)) {
        matches.push({ conversation: conv, matchType: "title", snippet: conv.title || "Untitled" })
        continue
      }
      const storeConv = storeConversations.find((c) => c.id === conv.id)
      if (storeConv) {
        const msg = storeConv.messages.find((m) => m.content.toLowerCase().includes(lower))
        if (msg) {
          matches.push({
            conversation: conv,
            matchType: "message",
            snippet: getSnippet(msg.content, q),
          })
        }
      }
    }
    return matches
  }, [query, conversations, storeConversations])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset selection when results change
    setSelectedIdx(0)
  }, [results])

  const handleSelect = useCallback((id: string) => {
    onSelect(id)
    onOpenChange(false)
  }, [onSelect, onOpenChange])

  function handleKeyDown(e: React.KeyboardEvent) {
    if (results.length === 0) return

    if (e.key === "ArrowDown") {
      e.preventDefault()
      setSelectedIdx((i) => Math.min(i + 1, results.length - 1))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setSelectedIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === "Enter") {
      e.preventDefault()
      const r = results[selectedIdx]
      if (r) handleSelect(r.conversation.id)
    }
  }

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const selected = el.querySelector(`[data-idx="${selectedIdx}"]`)
    if (selected) {
      selected.scrollIntoView({ block: "nearest" })
    }
  }, [selectedIdx])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="!max-w-[580px] !h-[420px] flex flex-col p-0 !gap-0 overflow-hidden rounded-xl shadow-2xl border-border/80"
        showCloseButton={false}
        onKeyDown={handleKeyDown}
      >
        <DialogTitle className="sr-only">Search conversations</DialogTitle>

        {/* Input — fixed at top */}
        <div className="flex items-center gap-3 px-4 shrink-0 border-b border-border/40">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground/50" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations..."
            className="flex-1 h-12 bg-transparent text-sm outline-none placeholder:text-muted-foreground/40"
          />
          {query && results.length > 0 && (
            <span className="shrink-0 inline-flex items-center rounded-full bg-rh-blue-light/50 dark:bg-rh-blue-dark/40 px-2 py-0.5 text-[11px] font-medium text-primary dark:text-primary">
              {results.length}
            </span>
          )}
        </div>

        {/* Results — fills remaining space, scrolls internally */}
        <div ref={listRef} className="flex-1 min-h-0 overflow-y-auto overscroll-contain">
          {query.trim() === "" ? (
            <div className="flex flex-col items-center justify-center h-full gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted/50">
                <Search className="h-5 w-5 text-muted-foreground/30" />
              </div>
              <p className="text-[13px] text-muted-foreground/50">Search by title or message content</p>
            </div>
          ) : results.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-2">
              <p className="text-[13px] text-muted-foreground">No matches for &ldquo;{query}&rdquo;</p>
              <p className="text-xs text-muted-foreground/40">Try a different keyword</p>
            </div>
          ) : (
            <div className="py-1.5 px-1.5">
              {results.map((result, i) => {
                const isSelected = i === selectedIdx
                return (
                  <button
                    key={result.conversation.id}
                    data-idx={i}
                    onClick={() => handleSelect(result.conversation.id)}
                    onMouseEnter={() => setSelectedIdx(i)}
                    className={cn(
                      "w-full flex items-start gap-3 px-3 py-2.5 text-left rounded-lg transition-all duration-150",
                      isSelected
                        ? "bg-accent shadow-sm border-l-2 border-l-primary pl-[10px]"
                        : "border-l-2 border-l-transparent",
                    )}
                  >
                    <div className={cn(
                      "flex h-7 w-7 shrink-0 items-center justify-center rounded-md mt-0.5 transition-colors duration-150",
                      isSelected
                        ? "bg-rh-blue-light text-primary dark:bg-rh-blue-dark/60 dark:text-primary"
                        : "bg-muted/50 text-muted-foreground/50",
                    )}>
                      {result.matchType === "title"
                        ? <MessageSquare className="h-3.5 w-3.5" />
                        : <Quote className="h-3.5 w-3.5" />
                      }
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={cn(
                        "text-[13px] font-medium truncate transition-colors duration-150",
                        isSelected ? "text-foreground" : "text-foreground/70",
                      )}>
                        {result.conversation.title || "Untitled"}
                      </p>
                      <p className="text-xs text-muted-foreground/60 mt-0.5 line-clamp-1">
                        {result.matchType === "title" ? (
                          "Title match"
                        ) : (
                          <HighlightedSnippet text={result.snippet} query={query} />
                        )}
                      </p>
                    </div>
                    <CornerDownLeft className={cn(
                      "h-3.5 w-3.5 shrink-0 mt-1.5 transition-opacity duration-150",
                      isSelected ? "opacity-40" : "opacity-0",
                    )} />
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer — fixed at bottom */}
        <div className="flex items-center gap-4 px-4 py-2 shrink-0 border-t border-border/40 bg-muted/15">
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground/40">
            <kbd className="inline-flex h-4 min-w-[16px] items-center justify-center rounded border border-border/50 bg-background px-1 font-mono text-[10px] text-muted-foreground/50">↑</kbd>
            <kbd className="inline-flex h-4 min-w-[16px] items-center justify-center rounded border border-border/50 bg-background px-1 font-mono text-[10px] text-muted-foreground/50">↓</kbd>
            navigate
          </span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground/40">
            <kbd className="inline-flex h-4 min-w-[16px] items-center justify-center rounded border border-border/50 bg-background px-1 font-mono text-[10px] text-muted-foreground/50">↵</kbd>
            open
          </span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground/40">
            <kbd className="inline-flex h-4 min-w-[16px] items-center justify-center rounded border border-border/50 bg-background px-1 font-mono text-[10px] text-muted-foreground/50">esc</kbd>
            close
          </span>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function HighlightedSnippet({ text, query }: { text: string; query: string }) {
  const lower = text.toLowerCase()
  const qLower = query.toLowerCase()
  const idx = lower.indexOf(qLower)
  if (idx === -1) return <>{text}</>

  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-rh-blue-light/80 dark:bg-rh-blue/25 text-rh-blue-dark dark:text-rh-blue-light rounded-sm px-0.5 font-medium">
        {text.slice(idx, idx + query.length)}
      </mark>
      {text.slice(idx + query.length)}
    </>
  )
}
