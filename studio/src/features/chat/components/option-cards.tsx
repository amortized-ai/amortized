import { useState, useRef, useEffect } from "react"
import { ArrowRight, PenLine, CircleCheck } from "lucide-react"
import { cn } from "@/lib/utils"
import type { OptionCard } from "../types"

interface OptionCardsProps {
  cards: OptionCard[]
  onSelect: (value: string) => void
  showOther?: boolean
  selectedValue?: string
}

export function OptionCards({ cards, onSelect, showOther = true, selectedValue }: OptionCardsProps) {
  const [otherOpen, setOtherOpen] = useState(false)
  const [otherValue, setOtherValue] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)
  const locked = selectedValue != null

  useEffect(() => {
    if (otherOpen) inputRef.current?.focus()
  }, [otherOpen])

  function handleOtherSubmit() {
    const trimmed = otherValue.trim()
    if (!trimmed) return
    onSelect(trimmed)
    setOtherValue("")
    setOtherOpen(false)
  }

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {cards.map((card, i) => {
          const isSelected = locked && card.value === selectedValue
          const isUnselected = locked && card.value !== selectedValue

          return (
            <button
              key={card.value}
              className={cn(
                "animate-message-in group rounded-xl border p-3 text-left transition-all duration-300",
                isSelected
                  ? "border-l-[3px] border-l-rh-green bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                  : isUnselected
                    ? "border-l-[3px] border-l-border/30 opacity-50"
                    : "border-l-[3px] border-l-primary/60 dark:border-l-primary bg-card hover:border-l-primary hover:bg-rh-blue-light/50 dark:hover:bg-rh-blue-dark/30 hover:shadow-md hover:-translate-y-0.5",
                locked && "pointer-events-none",
              )}
              style={{ animationDelay: `${i * 150}ms` }}
              onClick={() => onSelect(card.value)}
              tabIndex={locked ? -1 : 0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  onSelect(card.value)
                }
              }}
            >
              <div className="flex items-center gap-2">
                {isSelected && (
                  <CircleCheck className="h-4 w-4 shrink-0 text-primary" />
                )}
                <span className={cn(
                  "flex-1 text-sm font-medium",
                  isUnselected && "text-muted-foreground",
                )}>{card.title}</span>
                {!locked && (
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/40 transition-all duration-300 group-hover:text-primary group-hover:translate-x-0.5" />
                )}
              </div>
              {card.description && (
                <p className={cn(
                  "mt-1 text-xs text-muted-foreground",
                  isUnselected && "text-muted-foreground/60",
                )}>{card.description}</p>
              )}
            </button>
          )
        })}
        {showOther && !locked && !otherOpen && (
          <button
            className="animate-message-in group rounded-xl border border-dashed bg-card/50 p-3 text-left transition-all duration-300 hover:border-primary/40 hover:bg-card hover:shadow-md hover:-translate-y-0.5 dark:hover:border-primary/50"
            style={{ animationDelay: `${cards.length * 150}ms` }}
            onClick={() => setOtherOpen(true)}
          >
            <div className="flex items-center gap-2">
              <PenLine className="h-3.5 w-3.5 text-muted-foreground/60" />
              <span className="flex-1 text-sm font-medium text-muted-foreground">Other</span>
            </div>
          </button>
        )}
      </div>
      {otherOpen && !locked && (
        <div className="animate-message-in flex items-center gap-2 rounded-xl border bg-card p-2">
          <input
            ref={inputRef}
            type="text"
            value={otherValue}
            onChange={(e) => setOtherValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                handleOtherSubmit()
              }
              if (e.key === "Escape") {
                setOtherOpen(false)
                setOtherValue("")
              }
            }}
            placeholder="Type your answer..."
            className="flex-1 bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground/50"
          />
          <button
            onClick={handleOtherSubmit}
            disabled={!otherValue.trim()}
            className="shrink-0 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity disabled:opacity-40"
          >
            Send
          </button>
        </div>
      )}
    </div>
  )
}
