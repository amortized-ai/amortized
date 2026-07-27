import { useState, useRef, useEffect } from "react"
import { Pencil } from "lucide-react"
import { cn } from "@/lib/utils"

interface EditableTitleProps {
  value: string
  onSave: (newValue: string) => void | Promise<void>
  className?: string
  inputClassName?: string
}

export function EditableTitle({ value, onSave, className, inputClassName }: EditableTitleProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  const [saving, setSaving] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [editing])

  function startEditing() {
    setDraft(value)
    setEditing(true)
  }

  async function commit() {
    const trimmed = draft.trim()
    if (!trimmed || trimmed === value) {
      setEditing(false)
      return
    }
    setSaving(true)
    try {
      await onSave(trimmed)
    } finally {
      setSaving(false)
      setEditing(false)
    }
  }

  function cancel() {
    setDraft(value)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1.5 min-w-0">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void commit()
            if (e.key === "Escape") cancel()
          }}
          onBlur={() => void commit()}
          disabled={saving}
          className={cn(
            "min-w-0 flex-1 bg-transparent outline-none border-b-2 border-primary/60 dark:border-primary/50 py-0.5 text-base font-semibold",
            inputClassName,
          )}
        />
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={startEditing}
      className={cn(
        "group/edit flex items-center gap-1.5 min-w-0 rounded-md -mx-1 px-1 transition-colors hover:bg-muted/50",
        className,
      )}
      title="Click to rename"
    >
      <span className="truncate">{value}</span>
      <Pencil className="h-3 w-3 shrink-0 text-muted-foreground/0 transition-colors group-hover/edit:text-muted-foreground/60" />
    </button>
  )
}
