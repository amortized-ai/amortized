import { Search, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

export function SearchInput({ value, onChange, placeholder = "Search...", className }: SearchInputProps) {
  return (
    <div
      className={cn(
        "group/search flex items-center gap-2 h-9 rounded-lg border px-3",
        "bg-muted/30 border-border/50 shadow-none",
        "hover:bg-muted/50 hover:border-border/70",
        "focus-within:bg-background focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/15",
        "dark:bg-muted/20 dark:border-border/40 dark:hover:bg-muted/30 dark:hover:border-border/60",
        "dark:focus-within:bg-background dark:focus-within:border-primary/50 dark:focus-within:ring-primary/10",
        "transition-all duration-200",
        "w-56 sm:w-64",
        className,
      )}
    >
      <Search className="size-3.5 shrink-0 text-muted-foreground/60 transition-colors duration-200 group-focus-within/search:text-primary/70" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="flex-1 min-w-0 bg-transparent text-[13px] outline-none placeholder:text-muted-foreground/50"
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="shrink-0 flex items-center justify-center size-5 rounded text-muted-foreground/50 hover:text-foreground hover:bg-muted/60 transition-colors duration-150"
        >
          <X className="size-3" />
        </button>
      )}
    </div>
  )
}
